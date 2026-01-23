"""
VoxCPM TTS Engine Implementation

Tokenizer-free TTS engine with voice cloning support using VoxCPM from OpenBMB.
"""

import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from app.engines.base import BaseTTSEngine
from app.engines.tts.voxcpm.config import VoxCPMConfig
from app.exceptions import EngineNotReadyError, SynthesisError
from app.models.engine import TTSChunk, TTSResponse
from app.models.metrics import TTSPerformanceMetrics


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
        **kwargs,
    ) -> TTSResponse:
        """
        Synthesize text to speech (batch mode).

        Args:
            text: Text to synthesize
            voice: Not used (voice cloning via kwargs instead)
            speed: Not directly supported by VoxCPM
            **kwargs: Additional parameters:
                - prompt_wav_path: Path to reference audio for voice cloning
                - prompt_text: Transcript of reference audio

        Returns:
            TTSResponse with audio data and metrics
        """
        start_time = time.time()

        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("VoxCPM model not loaded")

        # Extract voice cloning parameters
        prompt_wav_path = kwargs.get("prompt_wav_path")
        prompt_text = kwargs.get("prompt_text")

        # Validate prompt audio if provided
        if prompt_wav_path is not None:
            prompt_path = Path(prompt_wav_path)
            if not prompt_path.exists():
                raise SynthesisError(f"Prompt audio file not found: {prompt_wav_path}")

        processing_start = time.time()
        try:
            # Generate audio using VoxCPM
            wav = self._model.generate(
                text=text,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
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
            audio_bytes = self._numpy_to_wav_bytes(wav, sample_rate)

            # Calculate duration
            duration_seconds = len(wav) / sample_rate

        except Exception as e:
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
        **kwargs,
    ) -> AsyncIterator[TTSChunk | TTSResponse]:
        """
        Streaming synthesis - yields audio chunks progressively.

        Args:
            text: Text to synthesize
            **kwargs: Additional parameters (same as synthesize)

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

        # Extract voice cloning parameters
        prompt_wav_path = kwargs.get("prompt_wav_path")
        prompt_text = kwargs.get("prompt_text")

        # Validate prompt audio if provided
        if prompt_wav_path is not None:
            prompt_path = Path(prompt_wav_path)
            if not prompt_path.exists():
                raise SynthesisError(f"Prompt audio file not found: {prompt_wav_path}")

        try:
            # Stream audio chunks
            for chunk in self._model.generate_streaming(
                text=text,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
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
                chunk_bytes = self._numpy_to_wav_bytes(chunk, sample_rate)

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
                audio_bytes = self._numpy_to_wav_bytes(full_audio, sample_rate)
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

    @staticmethod
    def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert numpy audio array to WAV bytes.

        Args:
            audio: Audio samples as numpy array (float32, range [-1, 1])
            sample_rate: Sample rate in Hz

        Returns:
            WAV file bytes
        """
        import io
        import wave

        # Normalize and convert to 16-bit PCM
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write to WAV format
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()
