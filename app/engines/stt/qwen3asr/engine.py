"""Qwen3-ASR STT Engine implementation using qwen-asr vLLM backend"""

import time
from collections.abc import AsyncIterator

from app.engines.base import BaseSTTEngine
from app.engines.stt.qwen3asr.config import Qwen3ASRConfig
from app.exceptions import (
    EngineNotReadyError,
    InvalidAudioError,
    TranscriptionError,
)
from app.models.engine import Segment, STTChunk, STTResponse
from app.models.metrics import STTPerformanceMetrics
from app.types.audio import AudioInput
from app.utils.audio import AudioProcessor


class Qwen3ASREngine(BaseSTTEngine):
    """
    Qwen3-ASR STT Engine using vLLM backend

    High-accuracy multilingual speech recognition supporting 52 languages.
    Uses the qwen-asr package with vLLM backend for fast inference.

    Features:
    - State-of-the-art accuracy
    - 52 language/dialect support
    - Automatic language detection
    - Streaming and offline inference
    - Long audio support (up to 20 minutes)
    """

    def __init__(self, config: Qwen3ASRConfig):
        """
        Initialize Qwen3-ASR STT engine

        Args:
            config: Qwen3ASRConfig with vLLM-specific parameters
        """
        super().__init__(config)
        self.qwen_config = config
        self._model = None
        self._audio_processor = AudioProcessor()

    async def _initialize(self) -> None:
        """Load Qwen3-ASR model using vLLM backend"""
        try:
            from qwen_asr import Qwen3ASRModel

            # Initialize with vLLM backend via LLM constructor
            self._model = Qwen3ASRModel.LLM(
                model=self.qwen_config.model_name,
                gpu_memory_utilization=self.qwen_config.gpu_memory_utilization,
                max_inference_batch_size=self.qwen_config.max_inference_batch_size,
                max_new_tokens=self.qwen_config.max_new_tokens,
                forced_aligner=self.qwen_config.forced_aligner,
            )
        except ImportError as e:
            raise EngineNotReadyError(
                "qwen-asr package not installed. Run: pip install qwen-asr[vllm]"
            ) from e
        except Exception as e:
            raise EngineNotReadyError(f"Failed to load Qwen3-ASR model: {e}") from e

    async def _cleanup(self) -> None:
        """Cleanup Qwen3-ASR model resources"""
        self._model = None

    async def transcribe(
        self, audio_data: AudioInput, language: str | None = None, **kwargs
    ) -> STTResponse:
        """
        Transcribe audio to text (batch mode)

        Args:
            audio_data: Audio in bytes, numpy, Path, or BytesIO format
            language: Optional language hint (e.g., "English", "Chinese", "Vietnamese")
                     None for automatic detection
            **kwargs: Additional parameters

        Returns:
            STTResponse with text, language, and performance metrics

        Raises:
            InvalidAudioError: If audio is empty or invalid
            TranscriptionError: If transcription fails
            EngineNotReadyError: If engine not initialized
        """
        start_time = time.time()

        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("Qwen3-ASR model not loaded")

        # Convert audio to numpy array
        audio_array, sample_rate = self._audio_processor.to_numpy(audio_data)

        # Validate audio
        if len(audio_array) == 0:
            raise InvalidAudioError("Audio is empty")

        # Get audio duration
        audio_duration_ms = self._audio_processor.get_duration_ms(
            audio_array, sample_rate
        )

        # Transcribe with Qwen3-ASR
        processing_start = time.time()

        try:
            # Prepare language (map short codes to full names if needed)
            lang = self._map_language(language or self.qwen_config.language)

            # qwen-asr accepts (np.ndarray, sample_rate) tuple
            results = self._model.transcribe(
                audio=(audio_array, sample_rate),
                language=lang,
                return_time_stamps=(self.qwen_config.forced_aligner is not None),
            )

            # Extract result (batch size = 1)
            results_segments = None
            if results and len(results) > 0:
                result = results[0]

                text = result.text if hasattr(result, "text") else str(result)
                detected_language = (
                    result.language if hasattr(result, "language") else lang
                )

                # Extract timestamps if available (requires forced_aligner)
                # Note: Qwen3-ASR returns 'time_stamps' attribute containing ForcedAlignResult (list of ForcedAlignItem)
                if hasattr(result, "time_stamps") and result.time_stamps:
                    results_segments = []
                    for seg in result.time_stamps:
                        results_segments.append(
                            Segment(
                                start=float(getattr(seg, "start_time", 0.0)),
                                end=float(getattr(seg, "end_time", 0.0)),
                                text=str(getattr(seg, "text", "")),
                            )
                        )
            else:
                text = ""
                detected_language = lang

        except Exception as e:
            raise TranscriptionError(f"Qwen3-ASR transcription failed: {e}") from e

        processing_end = time.time()
        end_time = time.time()

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

        return STTResponse(
            text=text,
            language=detected_language,
            segments=results_segments,
            performance_metrics=metrics,
        )

    async def transcribe_stream(
        self, audio_data: AudioInput, language: str | None = None, **kwargs
    ) -> AsyncIterator[STTChunk | STTResponse]:
        """
        Transcribe audio in streaming mode

        For Qwen3-ASR, we simulate streaming by processing the audio
        and yielding the result as a single chunk, followed by final response.

        Args:
            audio_data: Audio input (bytes, numpy array, file path, or BytesIO)
            language: Optional language code
            **kwargs: Additional parameters

        Yields:
            STTChunk: Partial transcription chunk
            STTResponse: Final response with complete text and metrics
        """
        start_time = time.time()

        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("Qwen3-ASR model not loaded")

        try:
            # Convert audio to numpy array
            audio_array, sample_rate = self._audio_processor.to_numpy(audio_data)

            if len(audio_array) == 0:
                raise InvalidAudioError("Audio is empty")

            audio_duration_ms = self._audio_processor.get_duration_ms(
                audio_array, sample_rate
            )

            # Prepare language
            lang = self._map_language(language or self.qwen_config.language)

            # Process with Qwen3-ASR
            processing_start = time.time()
            results = self._model.transcribe(
                audio=(audio_array, sample_rate),
                language=lang,
                return_time_stamps=(self.qwen_config.forced_aligner is not None),
            )
            first_token_time = time.time()

            # Extract result
            if results and len(results) > 0:
                result = results[0]
                text = result.text if hasattr(result, "text") else str(result)
                detected_language = (
                    result.language if hasattr(result, "language") else lang
                )
            else:
                text = ""
                detected_language = lang

            # Yield single chunk (simulated streaming)
            chunk_latency_ms = (time.time() - processing_start) * 1000
            yield STTChunk(
                text=text,
                timestamp=0.0,
                confidence=None,
                chunk_latency_ms=chunk_latency_ms,
            )

            # Calculate final metrics
            end_time = time.time()
            total_duration_ms = (end_time - start_time) * 1000
            time_to_first_token_ms = (first_token_time - start_time) * 1000
            processing_time_ms = (first_token_time - processing_start) * 1000

            metrics = STTPerformanceMetrics(
                latency_ms=total_duration_ms,
                processing_time_ms=processing_time_ms,
                audio_duration_ms=audio_duration_ms,
                real_time_factor=(
                    processing_time_ms / audio_duration_ms
                    if audio_duration_ms > 0
                    else None
                ),
                time_to_first_token_ms=time_to_first_token_ms,
                total_stream_duration_ms=total_duration_ms,
                total_chunks=1,
            )

            # Extract timestamps for streaming response
            results_segments = None

            if hasattr(result, "time_stamps") and result.time_stamps:
                results_segments = []
                for seg in result.time_stamps:
                    results_segments.append(
                        Segment(
                            start=float(getattr(seg, "start_time", 0.0)),
                            end=float(getattr(seg, "end_time", 0.0)),
                            text=str(getattr(seg, "text", "")),
                        )
                    )

            # Yield final response
            yield STTResponse(
                text=text,
                language=detected_language,
                segments=results_segments,
                performance_metrics=metrics,
            )

        except Exception as e:
            if isinstance(
                e, (InvalidAudioError, TranscriptionError, EngineNotReadyError)
            ):
                raise
            raise TranscriptionError(f"Qwen3-ASR stream failed: {e}") from e

    def _map_language(self, language: str | None) -> str | None:
        """
        Map short language codes to Qwen3-ASR language names

        Qwen3-ASR uses full language names like "English", "Chinese", "Vietnamese"
        """
        if language is None:
            return None

        # Common mappings
        lang_map = {
            "en": "English",
            "zh": "Chinese",
            "vi": "Vietnamese",
            "ja": "Japanese",
            "ko": "Korean",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ar": "Arabic",
            "th": "Thai",
            "id": "Indonesian",
            "ms": "Malay",
        }

        return lang_map.get(language.lower(), language)

    @property
    def supported_formats(self) -> list[str]:
        """List of supported audio formats"""
        return ["wav", "mp3", "flac", "ogg", "m4a", "opus"]

    @property
    def engine_name(self) -> str:
        """Engine name for identification"""
        return "qwen3-asr"
