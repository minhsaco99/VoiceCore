"""Unit tests for TTS router endpoints"""

import base64


class TestSynthesizeEndpoint:
    """POST /synthesize tests"""

    def test_returns_200_with_valid_text(self, client_both):
        """Returns 200 with valid text and audio data"""
        response = client_both.post(
            "/api/v1/tts/synthesize",
            params={"engine": "default"},
            data={"text": "Hello world"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "audio_data" in data
        assert "sample_rate" in data
        assert "duration_seconds" in data
        assert "format" in data

        # Verify audio is base64 encoded
        audio_bytes = base64.b64decode(data["audio_data"])
        assert len(audio_bytes) > 0

    def test_returns_422_when_missing_text(self, client_both):
        """Returns 422 when text param missing"""
        response = client_both.post(
            "/api/v1/tts/synthesize",
            params={"engine": "default"},
        )

        assert response.status_code == 422

    def test_returns_404_when_engine_not_found(self, client_both):
        """Returns 404 when engine not found"""
        response = client_both.post(
            "/api/v1/tts/synthesize",
            params={"engine": "nonexistent"},
            data={"text": "Hello"},
        )

        assert response.status_code == 404

    def test_returns_400_for_invalid_json_params(self, client_both):
        """Returns 400 for invalid JSON in engine_params"""
        response = client_both.post(
            "/api/v1/tts/synthesize",
            params={"engine": "default"},
            data={
                "text": "Hello",
                "engine_params": "invalid json{",
            },
        )

        assert response.status_code == 400
        assert "Invalid engine_params JSON" in response.json()["detail"]

    def test_passes_valid_engine_params(self, client_both, mock_tts_engine):
        """Passes valid engine_params to engine"""
        response = client_both.post(
            "/api/v1/tts/synthesize",
            params={"engine": "default"},
            data={
                "text": "Hello",
                "engine_params": '{"pitch": 1.5}',
            },
        )

        assert response.status_code == 200
        # Verify engine was called with params
        mock_tts_engine.synthesize.assert_called_once()
        call_kwargs = mock_tts_engine.synthesize.call_args.kwargs
        assert call_kwargs.get("pitch") == 1.5

    def test_passes_voice_and_speed_params(self, client_both, mock_tts_engine):
        """Passes voice and speed params to engine"""
        response = client_both.post(
            "/api/v1/tts/synthesize",
            params={
                "engine": "default",
                "voice": "voice2",
                "speed": 1.5,
            },
            data={"text": "Hello"},
        )

        assert response.status_code == 200
        call_kwargs = mock_tts_engine.synthesize.call_args.kwargs
        assert call_kwargs.get("voice") == "voice2"
        assert call_kwargs.get("speed") == 1.5

    def test_passes_reference_audio_and_text(self, client_both, mock_tts_engine):
        """Passes reference_audio and reference_text to engine for voice cloning"""
        response = client_both.post(
            "/api/v1/tts/synthesize",
            params={"engine": "default"},
            data={
                "text": "Hello",
                "reference_text": "Reference transcript",
            },
            files={"reference_audio": ("ref.wav", b"fake audio data", "audio/wav")},
        )

        assert response.status_code == 200
        call_kwargs = mock_tts_engine.synthesize.call_args.kwargs
        assert call_kwargs.get("reference_audio") == b"fake audio data"
        assert call_kwargs.get("reference_text") == "Reference transcript"


class TestSynthesizeStreamEndpoint:
    """POST /synthesize/stream tests"""

    def test_returns_200_with_event_stream(self, client_both):
        """Returns 200 with event-stream content type"""
        response = client_both.post(
            "/api/v1/tts/synthesize/stream",
            params={"engine": "default"},
            data={"text": "Hello"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_streams_chunks_and_complete(self, client_both):
        """Streams chunk events and complete event"""
        response = client_both.post(
            "/api/v1/tts/synthesize/stream",
            params={"engine": "default"},
            data={"text": "Hello"},
        )

        # Parse SSE events
        events = []
        current_event = None
        for line in response.text.split("\n"):
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current_event:
                events.append(current_event)
                current_event = None

        assert "chunk" in events
        assert "complete" in events

    def test_returns_400_for_invalid_json_params(self, client_both):
        """Returns 400 for invalid JSON in engine_params"""
        response = client_both.post(
            "/api/v1/tts/synthesize/stream",
            params={"engine": "default"},
            data={
                "text": "Hello",
                "engine_params": "bad json",
            },
        )

        # Note: EventSourceResponse might handle errors differently,
        # but usually synchronous validation runs before streaming starts
        assert response.status_code == 400

    def test_passes_engine_params_stream(self, client_both, mock_tts_engine):
        """Passes engine_params to engine in stream mode"""
        response = client_both.post(
            "/api/v1/tts/synthesize/stream",
            params={"engine": "default"},
            data={
                "text": "Hello",
                "engine_params": '{"style": "happy"}',
            },
        )

        assert response.status_code == 200
        # Wait for generator to start (TestClient handles this)
        # Check call args
        mock_tts_engine.synthesize_stream.assert_called_once()
        call_kwargs = mock_tts_engine.synthesize_stream.call_args.kwargs
        assert call_kwargs.get("style") == "happy"

    def test_passes_reference_audio_stream(self, client_both, mock_tts_engine):
        """Passes reference_audio to engine in stream mode"""
        response = client_both.post(
            "/api/v1/tts/synthesize/stream",
            params={"engine": "default"},
            data={
                "text": "Hello",
                "reference_text": "Reference",
            },
            files={"reference_audio": ("ref.wav", b"audio bytes", "audio/wav")},
        )

        assert response.status_code == 200
        call_kwargs = mock_tts_engine.synthesize_stream.call_args.kwargs
        assert call_kwargs.get("reference_audio") == b"audio bytes"
        assert call_kwargs.get("reference_text") == "Reference"
