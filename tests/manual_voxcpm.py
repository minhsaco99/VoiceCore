import asyncio

from app.engines.tts.voxcpm.config import VoxCPMConfig
from app.engines.tts.voxcpm.engine import VoxCPMEngine


async def test_voxcpm():
    print("Initializing VoxCPM Engine...")
    config = VoxCPMConfig(model_name="openbmb/VoxCPM-0.5B", device="cuda")
    engine = VoxCPMEngine(config)

    try:
        await engine.initialize()
        print("Engine initialized successfully.")

        text = "Hello, this is a test of VoxCPM integration into VoiceCore."
        print(f"Synthesizing text: '{text}'")

        response = await engine.synthesize(text)
        print(f"Synthesis complete. Duration: {response.duration_seconds:.2f}s")
        print(f"Audio data size: {len(response.audio_data)} bytes")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(test_voxcpm())
