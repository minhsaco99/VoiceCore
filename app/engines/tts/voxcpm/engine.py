import asyncio
import io
import os
import tempfile
import time
from typing import AsyncIterator

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Monkey patch torchaudio.load to use soundfile directly
# This acts as a fallback/fix for environments where torchaudio 2.10 tries to use broken torchcodec
def safe_load(filepath, **kwargs):
    try:
        data, sampler_rate = sf.read(filepath)
        # Handle shape: soundfile returns (time, channels) or (time,)
        # torchaudio returns (channels, time)
        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0) # (time,) -> (1, time)
        else:
            tensor = tensor.transpose(0, 1) # (time, channels) -> (channels, time)
        
        return tensor.float(), sampler_rate
    except Exception as e:
        # Fallback to original if needed or re-raise
        raise RuntimeError(f"Failed to load audio with soundfile: {e}") from e

torchaudio.load = safe_load

from app.engines.base import BaseTTSEngine
from app.engines.tts.voxcpm.config import VoxCPMConfig
from app.exceptions import EngineNotReadyError, SynthesisError
from app.models.engine import TTSChunk, TTSResponse
from app.models.metrics import TTSPerformanceMetrics

# Type checking for VoxCPM library
try:
    from voxcpm import VoxCPM
except ImportError:
    VoxCPM = None

class VoxCPMEngine(BaseTTSEngine):
    """VoxCPM TTS engine implementation"""

    def __init__(self, config: VoxCPMConfig):
        super().__init__(config)
        self.vox_config = config
        self._model = None

    async def _initialize(self) -> None:
        """Load VoxCPM model"""
        if VoxCPM is None:
            raise ImportError("VoxCPM package not found. Please install with 'uv sync --group voxcpm'")
            
        # VoxCPM.from_pretrained can be slow and blocking, run in executor
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            None, 
            lambda: VoxCPM.from_pretrained(self.vox_config.model_name)
        )

    async def _cleanup(self) -> None:
        """Cleanup resources"""
        self._model = None

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,  # VoxCPM doesn't support speed control directly yet
        speaker_wav: bytes | None = None,
        **kwargs
    ) -> TTSResponse:
        """
        Synthesize text to speech (batch mode).
        """
        start_time = time.time()
        
        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("Model not loaded")

        processing_start = time.time()
        
        # Handle speaker_wav temp file
        temp_wav_path = None
        if speaker_wav:
             # Create temp file, close it so other processes can open it if needed
            fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.write(fd, speaker_wav)
            os.close(fd)

        try:
            # Prepare arguments
            generate_kwargs = {
                "text": text,
                "prompt_wav_path": temp_wav_path if temp_wav_path else kwargs.get("prompt_wav_path", self.vox_config.prompt_wav_path),
                "prompt_text": kwargs.get("prompt_text", self.vox_config.prompt_text),
                "cfg_value": kwargs.get("cfg_value", self.vox_config.cfg_value),
                "inference_timesteps": kwargs.get("inference_timesteps", self.vox_config.inference_timesteps),
                "normalize": kwargs.get("normalize", self.vox_config.normalize),
                "denoise": kwargs.get("denoise", self.vox_config.denoise),
                "retry_badcase": kwargs.get("retry_badcase", self.vox_config.retry_badcase),
                "retry_badcase_max_times": kwargs.get("retry_badcase_max_times", self.vox_config.retry_badcase_max_times),
            }

            # Run generation in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            wav = await loop.run_in_executor(
                None,
                lambda: self._model.generate(**generate_kwargs)
            )
            
            # Convert to bytes (assuming wav is float32 numpy array or similar)
            # We need to encode it to WAV bytes. The base class or external utils might handle this,
            # but usually TTSResponse expects bytes.
            # However, typically we prefer raw PCM or standard WAV.
            # Let's assume for now we return raw bytes or encoded WAV.
            # Since sf.write is used in example, wav is probably numpy array.
            
            buffer = io.BytesIO()
            sf.write(buffer, wav, self._model.tts_model.sample_rate, format='WAV')
            audio_data = buffer.getvalue()
            
            duration_seconds = len(wav) / self._model.tts_model.sample_rate

        except Exception as e:
            raise SynthesisError(f"Synthesis failed: {e}") from e
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

        processing_end = time.time()
        end_time = time.time()

        metrics = TTSPerformanceMetrics(
            latency_ms=(end_time - start_time) * 1000,
            processing_time_ms=(processing_end - processing_start) * 1000,
            audio_duration_ms=duration_seconds * 1000,
            real_time_factor=(
                ((processing_end - processing_start) * 1000) / (duration_seconds * 1000)
                if duration_seconds > 0 else None
            ),
            characters_processed=len(text),
        )

        return TTSResponse(
            audio_data=audio_data,
            sample_rate=self._model.tts_model.sample_rate,
            duration_seconds=duration_seconds,
            format="wav",
            performance_metrics=metrics,
        )

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        speaker_wav: bytes | None = None,
        **kwargs
    ) -> AsyncIterator[TTSChunk | TTSResponse]:
        """
        Streaming synthesis.
        """
        start_time = time.time()
        total_chunks = 0
        total_audio_array = []
        
        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("Model not loaded")

        # Handle speaker_wav temp file
        temp_wav_path = None
        if speaker_wav:
             # Create temp file
            fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.write(fd, speaker_wav)
            os.close(fd)

        try:
             # Prepare arguments
            generate_kwargs = {
                "text": text,
                # Streaming usually supports same args
                "prompt_wav_path": temp_wav_path if temp_wav_path else kwargs.get("prompt_wav_path", self.vox_config.prompt_wav_path),
                "prompt_text": kwargs.get("prompt_text", self.vox_config.prompt_text),
                "cfg_value": kwargs.get("cfg_value", self.vox_config.cfg_value),
                "inference_timesteps": kwargs.get("inference_timesteps", self.vox_config.inference_timesteps),
                "normalize": kwargs.get("normalize", self.vox_config.normalize),
                "denoise": kwargs.get("denoise", self.vox_config.denoise),
            }

            # Since generate_streaming is a generator, we need to iterate it.
            # But we can't iterate a standard generator in async loop without blocking.
            # Ideally we run this in a separate thread.
            
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()
            
            def producer():
                try:
                    for chunk in self._model.generate_streaming(**generate_kwargs):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    loop.call_soon_threadsafe(queue.put_nowait, None) # Sentinel
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, e)

            # Start producer in threaded executor
            loop.run_in_executor(None, producer)
            


            chunk_idx = 0
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                
                # item is likely a numpy array chunk
                chunk_start = time.time()
                total_chunks += 1
                total_audio_array.append(item)
                
                # Convert chunk to bytes? Or keep as numpy? 
                # API expects bytes usually. Raw PCM bytes is fast.
                # Assuming 16-bit PCM or float32 bytes.
                # Let's convert to bytes for transmission.
                audio_chunk_bytes = item.tobytes() 
                
                yield TTSChunk(
                    audio_data=audio_chunk_bytes,
                    sequence_number=chunk_idx,
                    chunk_latency_ms=(time.time() - start_time) * 1000, # Approximate
                )
                chunk_idx += 1

            # Final response
            end_time = time.time()
            full_audio = np.concatenate(total_audio_array)
            duration_seconds = len(full_audio) / self._model.tts_model.sample_rate
            
            # Export full wav
            buffer = io.BytesIO()
            sf.write(buffer, full_audio, self._model.tts_model.sample_rate, format='WAV')
            full_audio_bytes = buffer.getvalue()

            metrics = TTSPerformanceMetrics(
                latency_ms=(end_time - start_time) * 1000,
                processing_time_ms=(end_time - start_time) * 1000,
                audio_duration_ms=duration_seconds * 1000,
                characters_processed=len(text),
                total_chunks=total_chunks,
            )

            yield TTSResponse(
                audio_data=full_audio_bytes,
                sample_rate=self._model.tts_model.sample_rate,
                duration_seconds=duration_seconds,
                format="wav",
                performance_metrics=metrics,
            )

        except Exception as e:
            raise SynthesisError(f"Stream synthesis failed: {e}") from e
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

    @property
    def supported_voices(self) -> list[str]:
        """List of available voices"""
        # VoxCPM is zero-shot, supports any voice via prompt. 
        # But maybe we can list a "default" one.
        return ["default"]   

    @property
    def engine_name(self) -> str:
        """Engine name"""
        return "voxcpm"
