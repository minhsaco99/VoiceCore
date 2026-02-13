"""
CosyVoice3 TTS Engine Implementation

HTTP client engine that communicates with an external CosyVoice3 FastAPI server.
Supports zero-shot voice cloning via prompt_wav + prompt_text.
"""

import io
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from app.engines.base import BaseTTSEngine
from app.engines.tts.cosyvoice3.config import CosyVoice3Config
from app.exceptions import EngineNotReadyError, SynthesisError
from app.models.engine import TTSChunk, TTSResponse
from app.models.metrics import TTSPerformanceMetrics
from app.utils.audio import temp_audio_file

logger = logging.getLogger(__name__)


class CosyVoice3Engine(BaseTTSEngine):
    """
    CosyVoice3 TTS Engine (HTTP Client)

    Connects to an external CosyVoice3 FastAPI server for zero-shot TTS synthesis.
    The server streams raw PCM int16 audio at the configured sample rate.

    Supported server endpoints:
    - POST /inference_zero_shot: tts_text + prompt_text + speed + prompt_wav file

    Features:
    - Zero-shot voice cloning with reference audio
    - Streaming synthesis support
    - Voice map configuration for preset voices
    - Persistent HTTP connection with keep-alive
    """

    def __init__(self, config: CosyVoice3Config):
        super().__init__(config)
        self.cv3_config = config
        self._client = None

    async def _initialize(self) -> None:
        """
        Create persistent HTTP client and verify server connectivity.
        """
        try:
            import httpx
        except ImportError as e:
            raise EngineNotReadyError(
                "httpx package not installed. Run: uv sync --group cosyvoice3"
            ) from e

        timeout = httpx.Timeout(
            connect=self.cv3_config.connect_timeout,
            read=self.cv3_config.read_timeout,
            write=self.cv3_config.connect_timeout,
            pool=self.cv3_config.connect_timeout,
        )

        self._client = httpx.AsyncClient(
            base_url=self.cv3_config.service_url,
            timeout=timeout,
        )

        # Verify server is reachable
        try:
            response = await self._client.get("/docs")
            logger.info(
                "CosyVoice3 server connected at %s (status: %d)",
                self.cv3_config.service_url,
                response.status_code,
            )
        except Exception as e:
            logger.warning(
                "CosyVoice3 server at %s may not be reachable: %s",
                self.cv3_config.service_url,
                e,
            )

    async def _cleanup(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _resolve_voice(
        self,
        voice: str | None,
        reference_audio: bytes | None,
        reference_text: str | None,
        **kwargs,
    ) -> tuple[str | None, str | None, bytes | None]:
        """
        Resolve voice to prompt_wav_path, prompt_text, and reference_audio bytes.

        Priority:
        1. Dynamic prompt_wav_path + prompt_text from kwargs
        2. reference_audio bytes (API-level voice cloning)
        3. Configured voice name
        4. Default voice

        Returns:
            (prompt_wav_path, prompt_text, reference_audio_bytes)
        """
        # 1. Dynamic path/text override from kwargs
        dyn_wav_path = kwargs.get("prompt_wav_path")
        dyn_prompt_text = kwargs.get("prompt_text")
        if dyn_wav_path and dyn_prompt_text:
            return dyn_wav_path, dyn_prompt_text, None

        # 2. Direct reference audio takes priority
        if reference_audio is not None:
            return None, reference_text, reference_audio

        # 3. Resolve voice name from config
        voice_name = voice or self.cv3_config.default_voice

        if voice_name is None:
            raise SynthesisError(
                "No voice specified and no default_voice configured. "
                f"Available voices: {list(self.cv3_config.voices.keys())}"
            )

        voice_config = self.cv3_config.voices.get(voice_name)
        if voice_config is None:
            raise SynthesisError(
                f"Voice '{voice_name}' not found. "
                f"Available: {list(self.cv3_config.voices.keys())}"
            )

        return voice_config.prompt_wav_path, voice_config.prompt_text, None

    def _prepare_prompt_text(self, prompt_text: str | None) -> str:
        """
        Prepare prompt_text with system prompt prefix for CosyVoice3.

        CosyVoice3 requires '<|endofprompt|>' marker in prompt_text.
        If not present, auto-prepend system_prompt + '<|endofprompt|>'.
        """
        if not prompt_text:
            return ""

        if "<|endofprompt|>" not in prompt_text:
            return f"{self.cv3_config.system_prompt}<|endofprompt|>{prompt_text}"

        return prompt_text

    async def _call_inference(
        self,
        tts_text: str,
        prompt_wav_path: str,
        prompt_text: str,
        speed: float,
    ) -> AsyncIterator[bytes]:
        """
        Call CosyVoice3 server /inference_zero_shot endpoint.

        Yields raw PCM int16 chunks from the streaming response.
        """
        if self._client is None:
            raise EngineNotReadyError("HTTP client not initialized")

        data = {
            "tts_text": tts_text,
            "prompt_text": prompt_text,
            "speed": float(speed),
        }

        try:
            with Path(prompt_wav_path).open("rb") as wav_file:
                files = {
                    "prompt_wav": (
                        "prompt_wav",
                        wav_file,
                        "application/octet-stream",
                    )
                }

                async with self._client.stream(
                    "POST",
                    "/inference_zero_shot",
                    data=data,
                    files=files,
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        if chunk:
                            yield chunk

        except Exception as e:
            if isinstance(e, (EngineNotReadyError, SynthesisError)):
                raise
            raise SynthesisError(f"CosyVoice3 server request failed: {e}") from e

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

        Calls CosyVoice3 server, collects all streaming PCM data,
        converts to WAV, and returns complete TTSResponse.

        Args:
            text: Text to synthesize
            voice: Voice name from config.voices (or None for default)
            speed: Speech speed multiplier
            reference_audio: Reference audio bytes for voice cloning (overrides voice)
            reference_text: Transcript of reference audio
            **kwargs: Additional parameters (ignored)

        Returns:
            TTSResponse with WAV audio data and metrics
        """
        start_time = time.time()

        await self._ensure_ready()

        if self._client is None:
            raise EngineNotReadyError("HTTP client not initialized")

        # Resolve voice
        prompt_wav_path, prompt_text, ref_audio_bytes = self._resolve_voice(
            voice, reference_audio, reference_text, **kwargs
        )
        prompt_text = self._prepare_prompt_text(prompt_text)
        effective_speed = speed if speed != 1.0 else self.cv3_config.speed

        # Use temp file if reference_audio bytes provided, otherwise use configured path
        with temp_audio_file(ref_audio_bytes) as temp_path:
            wav_path = temp_path or prompt_wav_path

            if wav_path is None:
                raise SynthesisError("No prompt WAV path available")

            processing_start = time.time()

            # Collect all PCM chunks
            pcm_data = b""
            async for chunk in self._call_inference(
                tts_text=text,
                prompt_wav_path=wav_path,
                prompt_text=prompt_text,
                speed=effective_speed,
            ):
                pcm_data += chunk

        processing_end = time.time()

        if len(pcm_data) == 0:
            raise SynthesisError("CosyVoice3 returned empty audio")

        # Convert PCM int16 to numpy, then to WAV bytes
        audio_array = (
            np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        audio_bytes = self._numpy_to_wav_bytes(audio_array, self.cv3_config.sample_rate)
        duration_seconds = len(audio_array) / self.cv3_config.sample_rate

        end_time = time.time()

        # Calculate metrics
        latency_ms = (end_time - start_time) * 1000
        processing_time_ms = (processing_end - processing_start) * 1000

        metrics = TTSPerformanceMetrics(
            latency_ms=latency_ms,
            processing_time_ms=processing_time_ms,
            real_time_factor=(
                processing_time_ms / (duration_seconds * 1000)
                if duration_seconds > 0
                else None
            ),
            characters_per_second=(
                len(text) / (processing_time_ms / 1000)
                if processing_time_ms > 0
                else None
            ),
        )

        return TTSResponse(
            audio_data=audio_bytes,
            sample_rate=self.cv3_config.sample_rate,
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

        Each chunk contains raw PCM data converted to WAV bytes.
        Final TTSResponse contains the complete concatenated audio.

        Args:
            text: Text to synthesize
            voice: Voice name from config.voices (or None for default)
            speed: Speech speed multiplier
            reference_audio: Reference audio bytes for voice cloning (overrides voice)
            reference_text: Transcript of reference audio
            **kwargs: Additional parameters (ignored)

        Yields:
            TTSChunk: Audio chunks with progressive generation
            TTSResponse: Final response with complete audio and metrics
        """
        start_time = time.time()
        first_chunk_time = None
        total_chunks = 0
        all_pcm_data = b""

        await self._ensure_ready()

        if self._client is None:
            raise EngineNotReadyError("HTTP client not initialized")

        # Resolve voice
        prompt_wav_path, prompt_text, ref_audio_bytes = self._resolve_voice(
            voice, reference_audio, reference_text, **kwargs
        )
        prompt_text = self._prepare_prompt_text(prompt_text)
        effective_speed = speed if speed != 1.0 else self.cv3_config.speed

        with temp_audio_file(ref_audio_bytes) as temp_path:
            wav_path = temp_path or prompt_wav_path

            if wav_path is None:
                raise SynthesisError("No prompt WAV path available")

            try:
                async for pcm_chunk in self._call_inference(
                    tts_text=text,
                    prompt_wav_path=wav_path,
                    prompt_text=prompt_text,
                    speed=effective_speed,
                ):
                    chunk_time = time.time()

                    if first_chunk_time is None:
                        first_chunk_time = chunk_time

                    all_pcm_data += pcm_chunk

                    # Convert PCM chunk to WAV bytes
                    chunk_array = (
                        np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    chunk_bytes = self._numpy_to_wav_bytes(
                        chunk_array, self.cv3_config.sample_rate
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

                if all_pcm_data:
                    full_array = (
                        np.frombuffer(all_pcm_data, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    audio_bytes = self._numpy_to_wav_bytes(
                        full_array, self.cv3_config.sample_rate
                    )
                    duration_seconds = len(full_array) / self.cv3_config.sample_rate
                else:
                    audio_bytes = b""
                    duration_seconds = 0.0

                total_duration_ms = (end_time - start_time) * 1000
                time_to_first_byte_ms = (
                    (first_chunk_time - start_time) * 1000 if first_chunk_time else None
                )

                metrics = TTSPerformanceMetrics(
                    latency_ms=total_duration_ms,
                    processing_time_ms=total_duration_ms,
                    real_time_factor=(
                        total_duration_ms / (duration_seconds * 1000)
                        if duration_seconds > 0
                        else None
                    ),
                    characters_per_second=(
                        len(text) / (total_duration_ms / 1000)
                        if total_duration_ms > 0
                        else None
                    ),
                    time_to_first_byte_ms=time_to_first_byte_ms,
                    total_stream_duration_ms=total_duration_ms,
                    total_chunks=total_chunks,
                )

                yield TTSResponse(
                    audio_data=audio_bytes,
                    sample_rate=self.cv3_config.sample_rate,
                    duration_seconds=duration_seconds,
                    format="wav",
                    performance_metrics=metrics,
                )

            except Exception as e:
                if isinstance(e, (EngineNotReadyError, SynthesisError)):
                    raise
                raise SynthesisError(f"CosyVoice3 streaming failed: {e}") from e

    @property
    def supported_voices(self) -> list[str]:
        """List of configured voice names."""
        return list(self.cv3_config.voices.keys()) or ["default"]

    @staticmethod
    def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    @property
    def engine_name(self) -> str:
        """Engine name for identification."""
        return "cosyvoice3"
