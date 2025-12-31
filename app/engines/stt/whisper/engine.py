"""Whisper STT Engine implementation using faster-whisper"""

import time
from collections.abc import AsyncIterator

from faster_whisper import WhisperModel

from app.engines.base import BaseSTTEngine
from app.engines.stt.whisper.config import WhisperConfig
from app.exceptions import (
    EngineNotReadyError,
    InvalidAudioError,
    TranscriptionError,
)
from app.models.engine import Segment, STTChunk, STTResponse
from app.models.metrics import STTPerformanceMetrics
from app.types.audio import AudioInput
from app.utils.audio import AudioProcessor


class WhisperSTTEngine(BaseSTTEngine):
    """
    Whisper STT engine implementation using faster-whisper

    Supports multiple audio input formats via AudioInput type alias.
    Provides accurate transcription with word-level timestamps.
    """

    def __init__(self, config: WhisperConfig):
        """
        Initialize Whisper STT engine

        Args:
            config: WhisperConfig with Whisper-specific parameters
        """
        super().__init__(config)
        self.whisper_config = config
        self._model: WhisperModel | None = None
        self._audio_processor = AudioProcessor()

    async def _initialize(self) -> None:
        """Load Whisper model"""
        self._model = WhisperModel(
            self.whisper_config.model_name,
            device=self.whisper_config.device,
            compute_type=self.whisper_config.compute_type,
        )

    async def _cleanup(self) -> None:
        """Cleanup Whisper model resources"""
        self._model = None

    async def transcribe(
        self, audio_data: AudioInput, language: str | None = None, **kwargs
    ) -> STTResponse:
        """
        Transcribe audio to text (invoke mode)

        Args:
            audio_data: Audio in bytes, numpy, Path, or BytesIO format
            language: Optional language hint (e.g., "en", "es")
            **kwargs: Additional Whisper parameters (overrides config defaults)
                     Examples: temperature, vad_filter, beam_size, etc.

        Returns:
            STTResponse with text, segments, and performance metrics

        Raises:
            InvalidAudioError: If audio is empty or invalid
            UnsupportedFormatError: If audio format not supported
            TranscriptionError: If transcription fails
            EngineNotReadyError: If engine not initialized
        """
        start_time = time.time()

        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("Whisper model not loaded")

        # Convert audio to numpy array
        audio_array, sample_rate = self._audio_processor.to_numpy(audio_data)

        # Validate audio
        if len(audio_array) == 0:
            raise InvalidAudioError("Audio is empty")

        # Resample to 16kHz for Whisper
        audio_16k = self._audio_processor.resample_to_16khz(audio_array, sample_rate)

        # Get audio duration
        audio_duration_ms = self._audio_processor.get_duration_ms(audio_16k, 16000)

        # Transcribe with Whisper
        processing_start = time.time()

        try:
            # Build parameters: config defaults + request overrides
            params = {
                "language": language or self.whisper_config.language,
                "beam_size": self.whisper_config.beam_size,
                "temperature": self.whisper_config.temperature,
                "vad_filter": self.whisper_config.vad_filter,
                "condition_on_previous_text": self.whisper_config.condition_on_previous_text,
                "compression_ratio_threshold": self.whisper_config.compression_ratio_threshold,
                "log_prob_threshold": self.whisper_config.log_prob_threshold,
                "no_speech_threshold": self.whisper_config.no_speech_threshold,
                "word_timestamps": True,  # Always enable for segments
            }

            # Override with request-specific params from engine_params
            params.update(kwargs)

            segments_iter, info = self._model.transcribe(audio_16k, **params)

            # Convert iterator to list
            segments_list = list(segments_iter)

        except Exception as e:
            raise TranscriptionError(f"Whisper transcription failed: {e}") from e

        processing_end = time.time()
        end_time = time.time()

        # Extract text from segments
        if not segments_list:
            text = ""
        else:
            # Join segment texts, handling both string and object types
            text_parts = []
            for seg in segments_list:
                if isinstance(seg, dict):
                    text_parts.append(seg.get("text", "").strip())
                elif hasattr(seg, "text"):
                    text_parts.append(seg.text.strip())
            text = " ".join(text_parts) if text_parts else ""

        # Build word-level segments
        word_segments = []
        for seg in segments_list:
            # Handle both dict and object types
            words = None
            if isinstance(seg, dict):
                words = seg.get("words")
            elif hasattr(seg, "words"):
                words = seg.words

            if words:
                for word in words:
                    # Extract word attributes
                    if isinstance(word, dict):
                        word_text = word.get("word", "")
                        word_start = word.get("start", 0.0)
                        word_end = word.get("end", 0.0)
                        confidence = word.get("probability")
                    else:
                        word_text = getattr(word, "word", "")
                        word_start = getattr(word, "start", 0.0)
                        word_end = getattr(word, "end", 0.0)
                        confidence = getattr(word, "probability", None)

                    if word_text:
                        word_segments.append(
                            Segment(
                                start=word_start,
                                end=word_end,
                                text=word_text.strip(),
                                confidence=confidence,
                            )
                        )

        # Calculate metrics
        latency_ms = (end_time - start_time) * 1000
        processing_time_ms = (processing_end - processing_start) * 1000
        real_time_factor = (
            processing_time_ms / audio_duration_ms if audio_duration_ms > 0 else None
        )

        metrics = STTPerformanceMetrics(
            latency_ms=latency_ms,
            processing_time_ms=processing_time_ms,
            audio_duration_ms=audio_duration_ms,
            real_time_factor=real_time_factor,
        )

        # Extract language from info
        language_detected = None
        if isinstance(info, dict):
            language_detected = info.get("language")
        elif hasattr(info, "language"):
            language_detected = info.language

        return STTResponse(
            text=text,
            language=language_detected,
            segments=word_segments if word_segments else None,
            performance_metrics=metrics,
        )

    async def transcribe_stream(
        self, audio_data: AudioInput, language: str | None = None, **kwargs
    ) -> AsyncIterator[STTChunk | STTResponse]:
        """
        Transcribe audio in streaming mode (progressive results)

        Processes audio and yields progressive transcription chunks (one per Whisper segment),
        followed by a final STTResponse with complete results.

        Args:
            audio_data: Audio input (bytes, numpy array, file path, or BytesIO)
            language: Optional language code (e.g., 'en', 'es')
            **kwargs: Additional Whisper parameters (overrides config defaults)
                     Examples: vad_filter, beam_size, temperature

        Yields:
            STTChunk: Partial transcription chunks (one per segment, is_final=False)
            STTChunk: Final marker chunk (is_final=True)
            STTResponse: Final response with complete text, segments, language, and metrics

        Raises:
            InvalidAudioError: If audio is invalid or empty
            TranscriptionError: If transcription fails
            EngineNotReadyError: If engine not initialized
        """
        # Initialize tracking variables
        start_time = time.time()
        first_token_time = None
        accumulated_text = []
        total_chunks = 0
        segments_list = []  # Track segments for final response

        # Ensure engine ready
        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("Whisper model not loaded")

        try:
            # Convert audio to numpy array (same as transcribe)
            audio_array, sample_rate = self._audio_processor.to_numpy(audio_data)

            # Validate audio
            if len(audio_array) == 0:
                raise InvalidAudioError("Audio is empty")

            # Resample to 16kHz for Whisper
            audio_16k = self._audio_processor.resample_to_16khz(
                audio_array, sample_rate
            )

            # Get audio duration
            audio_duration_ms = self._audio_processor.get_duration_ms(audio_16k, 16000)

            # Build parameters (config defaults + request overrides)
            params = {
                "language": language or self.whisper_config.language,
                "beam_size": self.whisper_config.beam_size,
                "temperature": self.whisper_config.temperature,
                "vad_filter": self.whisper_config.vad_filter,
                "condition_on_previous_text": self.whisper_config.condition_on_previous_text,
                "compression_ratio_threshold": self.whisper_config.compression_ratio_threshold,
                "log_prob_threshold": self.whisper_config.log_prob_threshold,
                "no_speech_threshold": self.whisper_config.no_speech_threshold,
                "word_timestamps": True,
            }

            # Update with kwargs (engine_params from request)
            params.update(kwargs)

            # Transcribe with Whisper (segments_iter is a generator)
            segments_iter, info = self._model.transcribe(audio_16k, **params)

            # Yield chunks progressively as segments are produced
            for segment in segments_iter:
                chunk_start_time = time.time()

                # Save segment for final response building
                segments_list.append(segment)

                # Extract text from segment (handle both object and dict types)
                if hasattr(segment, "text"):
                    text = segment.text.strip()
                elif isinstance(segment, dict):
                    text = segment.get("text", "").strip()
                else:
                    text = ""

                # Track first token time
                if first_token_time is None and text:
                    first_token_time = time.time()

                # Extract timestamp from segment
                timestamp = None
                if hasattr(segment, "start"):
                    timestamp = segment.start
                elif isinstance(segment, dict):
                    timestamp = segment.get("start")

                accumulated_text.append(text)
                total_chunks += 1

                # Calculate chunk latency
                chunk_latency_ms = (time.time() - chunk_start_time) * 1000

                # Yield STTChunk (not final)
                yield STTChunk(
                    text=text,
                    is_final=False,
                    timestamp=timestamp,
                    confidence=None,
                    chunk_latency_ms=chunk_latency_ms,
                )

            # Always yield final marker chunk (per user decision)
            yield STTChunk(
                text=accumulated_text[-1] if accumulated_text else "",
                is_final=True,
                timestamp=None,
                confidence=None,
                chunk_latency_ms=0.0,
            )
            total_chunks += 1  # Count final marker

            # Calculate final metrics
            end_time = time.time()
            total_duration_ms = (end_time - start_time) * 1000
            time_to_first_token_ms = (
                (first_token_time - start_time) * 1000 if first_token_time else None
            )

            # Build word-level segments from Whisper segments
            word_segments = []
            for segment in segments_list:
                words = (
                    segment.words if hasattr(segment, "words") else segment.get("words")
                )
                if words:
                    for word in words:
                        # Extract word attributes (handle both dict and object types)
                        if isinstance(word, dict):
                            word_text = word.get("word", "")
                            word_start = word.get("start", 0.0)
                            word_end = word.get("end", 0.0)
                            confidence = word.get("probability")
                        else:
                            word_text = getattr(word, "word", "")
                            word_start = getattr(word, "start", 0.0)
                            word_end = getattr(word, "end", 0.0)
                            confidence = getattr(word, "probability", None)

                        if word_text:
                            word_segments.append(
                                Segment(
                                    start=word_start,
                                    end=word_end,
                                    text=word_text.strip(),
                                    confidence=confidence,
                                )
                            )

            # Extract language from info
            language_detected = None
            if isinstance(info, dict):
                language_detected = info.get("language")
            elif hasattr(info, "language"):
                language_detected = info.language

            # Calculate metrics
            total_latency_ms = total_duration_ms  # End-to-end latency
            processing_time_ms = total_duration_ms  # Approximate (includes overhead)
            rtf = (
                processing_time_ms / audio_duration_ms
                if audio_duration_ms > 0
                else None
            )

            metrics = STTPerformanceMetrics(
                latency_ms=total_latency_ms,
                processing_time_ms=processing_time_ms,
                audio_duration_ms=audio_duration_ms,
                real_time_factor=rtf,
                # Streaming-specific metrics
                time_to_first_token_ms=time_to_first_token_ms,
                total_stream_duration_ms=total_duration_ms,
                total_chunks=total_chunks,
            )

            # Yield final STTResponse
            yield STTResponse(
                text=" ".join(accumulated_text),
                language=language_detected,
                segments=word_segments if word_segments else None,
                performance_metrics=metrics,
            )

        except Exception as e:
            if isinstance(
                e, (InvalidAudioError, TranscriptionError, EngineNotReadyError)
            ):
                raise
            raise TranscriptionError(f"Stream transcription failed: {e}") from e

    @property
    def supported_formats(self) -> list[str]:
        """List of supported audio formats"""
        return ["wav", "mp3", "flac", "ogg", "m4a", "opus"]

    @property
    def engine_name(self) -> str:
        """Engine name"""
        return "faster-whisper"
