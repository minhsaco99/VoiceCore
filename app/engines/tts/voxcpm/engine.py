"""
VoxCPM TTS Engine Implementation

Tokenizer-free TTS engine with voice cloning support using VoxCPM from OpenBMB.
"""

import time
from collections.abc import AsyncIterator

import numpy as np

from app.engines.base import BaseTTSEngine
from app.engines.tts.voxcpm.config import VoxCPMConfig
from app.exceptions import EngineNotReadyError, SynthesisError
from app.models.engine import TTSChunk, TTSResponse
from app.models.metrics import TTSPerformanceMetrics
from app.utils.audio import AudioProcessor, temp_audio_file


class VoxCPMEngine(BaseTTSEngine):
    """
    VoxCPM TTS Engine

    A tokenizer-free Text-to-Speech engine that models speech in continuous space.
    Supports context-aware speech generation and zero-shot voice cloning.

    Features:
    - High-quality speech synthesis
    - Zero-shot voice cloning with reference audio
    - Streaming synthesis support
    - Built on MiniCPM-4 backbone
    """

    def __init__(self, config: VoxCPMConfig):
        super().__init__(config)
        self.voxcpm_config = config
        self._model = None
        self.audio_processor = AudioProcessor()

    async def _initialize(self) -> None:
        """
        Initialize VoxCPM model.

        Loads the model from HuggingFace hub or local path.
        Note: VoxCPM automatically detects and uses available GPU.
        """
        try:
            from voxcpm import VoxCPM

            # VoxCPM auto-detects device (GPU if available)
            # load_denoiser=False since we handle denoising via config
            self._model = VoxCPM.from_pretrained(
                self.voxcpm_config.model_name,
                load_denoiser=self.voxcpm_config.denoise,
            )
        except ImportError as e:
            raise EngineNotReadyError(
                "VoxCPM package not installed. Run: pip install voxcpm"
            ) from e
        except Exception as e:
            raise EngineNotReadyError(f"Failed to load VoxCPM model: {e}") from e

    async def _cleanup(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            # Clear model reference to allow garbage collection
            self._model = None

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        reference_audio: bytes | None = None,
        reference_text: str | None = None,
        **kwargs,
    ) -> TTSResponse:
        """
        Synthesize text to speech (batch mode).

        Args:
            text: Text to synthesize
            voice: Not used (voice cloning via reference_audio instead)
            speed: Not directly supported by VoxCPM
            reference_audio: Reference audio bytes for voice cloning
            reference_text: Transcript of reference audio

        Returns:
            TTSResponse with audio data and metrics
        """
        start_time = time.time()

        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("VoxCPM model not loaded")

        with temp_audio_file(reference_audio) as prompt_wav_path:
            processing_start = time.time()
            try:
                # Generate audio using VoxCPM
                wav = self._model.generate(
                    text=text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=reference_text,
                    cfg_value=self.voxcpm_config.cfg_value,
                    inference_timesteps=self.voxcpm_config.inference_timesteps,
                    normalize=self.voxcpm_config.normalize,
                    denoise=self.voxcpm_config.denoise,
                    retry_badcase=self.voxcpm_config.retry_badcase,
                    retry_badcase_max_times=self.voxcpm_config.retry_badcase_max_times,
                    retry_badcase_ratio_threshold=self.voxcpm_config.retry_badcase_ratio_threshold,
                )

                # Get sample rate from model
                sample_rate = self._model.tts_model.sample_rate

                # Convert numpy array to bytes (16-bit PCM WAV)
                audio_bytes = self.audio_processor.numpy_to_wav_bytes(wav, sample_rate)

                # Calculate duration
                duration_seconds = len(wav) / sample_rate

            except Exception as e:
                if isinstance(e, (EngineNotReadyError, SynthesisError)):
                    raise
                raise SynthesisError(f"VoxCPM synthesis failed: {e}") from e

        processing_end = time.time()
        end_time = time.time()

        # Calculate metrics
        latency_ms = (end_time - start_time) * 1000
        processing_time_ms = (processing_end - processing_start) * 1000
        audio_duration_ms = duration_seconds * 1000

        metrics = TTSPerformanceMetrics(
            latency_ms=latency_ms,
            processing_time_ms=processing_time_ms,
            audio_duration_ms=audio_duration_ms,
            real_time_factor=(
                processing_time_ms / audio_duration_ms
                if audio_duration_ms > 0
                else None
            ),
            characters_processed=len(text),
        )

        return TTSResponse(
            audio_data=audio_bytes,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            format="wav",
            performance_metrics=metrics,
        )

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        reference_audio: bytes | None = None,
        reference_text: str | None = None,
        **kwargs,
    ) -> AsyncIterator[TTSChunk | TTSResponse]:
        """
        Streaming synthesis - yields audio chunks progressively.

        Args:
            text: Text to synthesize
            voice: Not used (voice cloning via reference_audio instead)
            speed: Not directly supported by VoxCPM
            reference_audio: Reference audio bytes for voice cloning
            reference_text: Transcript of reference audio

        Yields:
            TTSChunk: Audio chunks with progressive generation
            TTSResponse: Final response with complete audio and metrics
        """
        start_time = time.time()
        first_chunk_time = None
        total_chunks = 0
        all_audio_chunks = []

        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("VoxCPM model not loaded")

        with temp_audio_file(reference_audio) as prompt_wav_path:
            try:
                # Stream audio chunks
                for chunk in self._model.generate_streaming(
                    text=text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=reference_text,
                    cfg_value=self.voxcpm_config.cfg_value,
                    inference_timesteps=self.voxcpm_config.inference_timesteps,
                    normalize=self.voxcpm_config.normalize,
                    denoise=self.voxcpm_config.denoise,
                    retry_badcase=self.voxcpm_config.retry_badcase,
                    retry_badcase_max_times=self.voxcpm_config.retry_badcase_max_times,
                    retry_badcase_ratio_threshold=self.voxcpm_config.retry_badcase_ratio_threshold,
                ):
                    chunk_time = time.time()

                    if first_chunk_time is None:
                        first_chunk_time = chunk_time

                    # Store raw numpy chunk for final concatenation
                    all_audio_chunks.append(chunk)

                    # Get sample rate from model
                    sample_rate = self._model.tts_model.sample_rate

                    # Convert chunk to bytes
                    chunk_bytes = self.audio_processor.numpy_to_wav_bytes(
                        chunk, sample_rate
                    )

                    chunk_latency_ms = (chunk_time - start_time) * 1000

                    yield TTSChunk(
                        audio_data=chunk_bytes,
                        sequence_number=total_chunks,
                        chunk_latency_ms=chunk_latency_ms,
                    )

                    total_chunks += 1

                # Final response
                end_time = time.time()

                # Concatenate all chunks
                if all_audio_chunks:
                    full_audio = np.concatenate(all_audio_chunks)
                    sample_rate = self._model.tts_model.sample_rate
                    audio_bytes = self.audio_processor.numpy_to_wav_bytes(
                        full_audio, sample_rate
                    )
                    duration_seconds = len(full_audio) / sample_rate
                else:
                    audio_bytes = b""
                    sample_rate = 16000
                    duration_seconds = 0.0

                total_duration_ms = (end_time - start_time) * 1000
                time_to_first_byte_ms = (
                    (first_chunk_time - start_time) * 1000 if first_chunk_time else None
                )

                metrics = TTSPerformanceMetrics(
                    latency_ms=total_duration_ms,
                    processing_time_ms=total_duration_ms,
                    audio_duration_ms=duration_seconds * 1000,
                    real_time_factor=(
                        total_duration_ms / (duration_seconds * 1000)
                        if duration_seconds > 0
                        else None
                    ),
                    characters_processed=len(text),
                    time_to_first_byte_ms=time_to_first_byte_ms,
                    total_stream_duration_ms=total_duration_ms,
                    total_chunks=total_chunks,
                )

                yield TTSResponse(
                    audio_data=audio_bytes,
                    sample_rate=sample_rate,
                    duration_seconds=duration_seconds,
                    format="wav",
                    performance_metrics=metrics,
                )

            except Exception as e:
                if isinstance(e, (EngineNotReadyError, SynthesisError)):
                    raise
                raise SynthesisError(f"VoxCPM streaming failed: {e}") from e

    @property
    def supported_voices(self) -> list[str]:
        """
        List of supported voices.

        VoxCPM uses voice cloning instead of preset voices.
        Pass prompt_wav_path and prompt_text via kwargs for voice cloning.
        """
        return ["default"]

    @property
    def engine_name(self) -> str:
        """Engine name for identification."""
        return "voxcpm"
