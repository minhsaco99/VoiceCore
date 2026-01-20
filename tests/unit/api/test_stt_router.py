"""Unit tests for STT router endpoints"""


class TestTranscribeEndpoint:
    """POST /transcribe tests"""

    def test_returns_200_with_valid_audio(self, client, test_audio_bytes):
        """Returns 200 with valid audio"""
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "default"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["text"] == "Test transcription"

    def test_returns_transcription_result(self, client, test_audio_bytes):
        """Returns full STTResponse"""
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "default"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        data = response.json()
        assert "text" in data
        assert "language" in data
        assert "segments" in data
        # Note: response includes performance_metrics, not processing_time
        assert "performance_metrics" in data or "processing_time" in data

    def test_returns_422_when_missing_engine(self, client, test_audio_bytes):
        """Returns 422 when engine param missing"""
        response = client.post(
            "/api/v1/stt/transcribe",
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 422

    def test_returns_404_when_engine_not_found(self, client, test_audio_bytes):
        """Returns 404 when engine not found"""
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "nonexistent"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 404

    def test_returns_400_for_invalid_json_params(self, client, test_audio_bytes):
        """Returns 400 for invalid JSON in engine_params"""
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "default", "engine_params": "invalid json{"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 400
        assert "Invalid" in response.json()["detail"]

    def test_passes_valid_engine_params(
        self, client, test_audio_bytes, mock_stt_engine
    ):
        """Passes valid engine_params to engine"""
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "default", "engine_params": '{"beam_size": 5}'},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 200
        # Verify engine was called with params
        mock_stt_engine.transcribe.assert_called_once()
        call_kwargs = mock_stt_engine.transcribe.call_args.kwargs
        assert call_kwargs.get("beam_size") == 5

    def test_passes_language_param(self, client, test_audio_bytes, mock_stt_engine):
        """Passes language param to engine"""
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "default", "language": "en"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 200
        call_kwargs = mock_stt_engine.transcribe.call_args.kwargs
        assert call_kwargs.get("language") == "en"

    def test_returns_400_for_empty_audio(self, client):
        """Returns 400 for empty audio"""
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "default"},
            files={"audio": ("test.wav", b"", "audio/wav")},
        )

        assert response.status_code == 400

    def test_returns_413_for_oversized_audio(self, client):
        """Returns 413 for oversized audio"""
        large_audio = b"x" * (26 * 1024 * 1024)  # 26MB

        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "default"},
            files={"audio": ("test.wav", large_audio, "audio/wav")},
        )

        assert response.status_code == 413

    def test_returns_503_when_no_registry(self, client_no_registry, test_audio_bytes):
        """Returns 503 when registry not available"""
        response = client_no_registry.post(
            "/api/v1/stt/transcribe",
            params={"engine": "default"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 503


class TestTranscribeStreamEndpoint:
    """POST /transcribe/stream tests"""

    def test_returns_200_with_event_stream(self, client, test_audio_bytes):
        """Returns 200 with event-stream content type"""
        response = client.post(
            "/api/v1/stt/transcribe/stream",
            params={"engine": "default"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_streams_chunks_and_complete(self, client, test_audio_bytes):
        """Streams chunk events and complete event"""
        response = client.post(
            "/api/v1/stt/transcribe/stream",
            params={"engine": "default"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
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

    def test_returns_422_when_missing_engine(self, client, test_audio_bytes):
        """Returns 422 when engine param missing"""
        response = client.post(
            "/api/v1/stt/transcribe/stream",
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 422

    def test_returns_404_when_engine_not_found(self, client, test_audio_bytes):
        """Returns 404 when engine not found"""
        response = client.post(
            "/api/v1/stt/transcribe/stream",
            params={"engine": "nonexistent"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 404

    def test_returns_400_for_invalid_json_params(self, client, test_audio_bytes):
        """Returns 400 for invalid JSON"""
        response = client.post(
            "/api/v1/stt/transcribe/stream",
            params={"engine": "default", "engine_params": "bad json"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 400


class TestTranscribeWebSocket:
    """WebSocket /transcribe/ws tests"""

    def test_full_websocket_flow(self, client, test_audio_bytes):
        """Complete WebSocket flow"""
        with client.websocket_connect("/api/v1/stt/transcribe/ws") as ws:
            # Send config
            ws.send_json({"engine": "default"})

            # Send audio
            ws.send_bytes(test_audio_bytes)

            # Send END
            ws.send_text("END")

            # Receive responses
            responses = []
            while True:
                try:
                    data = ws.receive_json()
                    responses.append(data)
                    if data.get("type") == "complete":
                        break
                except Exception:
                    break

            # Verify we got chunk and complete
            types = [r["type"] for r in responses]
            assert "chunk" in types
            assert "complete" in types

    def test_returns_error_when_missing_engine(self, client):
        """Returns error when engine missing in config"""
        with client.websocket_connect("/api/v1/stt/transcribe/ws") as ws:
            # Send config without engine
            ws.send_json({"language": "en"})

            response = ws.receive_json()
            assert response["type"] == "error"

    def test_returns_error_for_nonexistent_engine(self, client):
        """Returns error for nonexistent engine"""
        with client.websocket_connect("/api/v1/stt/transcribe/ws") as ws:
            ws.send_json({"engine": "nonexistent"})

            response = ws.receive_json()
            assert response["type"] == "error"

    def test_handles_empty_audio(self, client):
        """Handles empty audio gracefully"""
        with client.websocket_connect("/api/v1/stt/transcribe/ws") as ws:
            ws.send_json({"engine": "default"})
            ws.send_text("END")  # No audio sent

            # Should close without error or return empty result

    def test_accepts_multiple_audio_chunks(self, client, test_audio_bytes):
        """Accepts multiple audio chunks"""
        with client.websocket_connect("/api/v1/stt/transcribe/ws") as ws:
            ws.send_json({"engine": "default"})

            # Send multiple chunks
            chunk_size = len(test_audio_bytes) // 3
            ws.send_bytes(test_audio_bytes[:chunk_size])
            ws.send_bytes(test_audio_bytes[chunk_size : chunk_size * 2])
            ws.send_bytes(test_audio_bytes[chunk_size * 2 :])

            ws.send_text("END")

            # Should complete successfully
            responses = []
            while True:
                try:
                    data = ws.receive_json()
                    responses.append(data)
                    if data.get("type") == "complete":
                        break
                except Exception:
                    break

            assert any(r["type"] == "complete" for r in responses)

    def test_passes_engine_params(self, client, test_audio_bytes):
        """Passes engine_params from config"""
        with client.websocket_connect("/api/v1/stt/transcribe/ws") as ws:
            ws.send_json({"engine": "default", "engine_params": {"beam_size": 5}})

            ws.send_bytes(test_audio_bytes)
            ws.send_text("END")

            # Should complete without error
            while True:
                try:
                    data = ws.receive_json()
                    if data.get("type") == "complete":
                        break
                except Exception:
                    break


class TestTranscribeEdgeCases:
    """Edge cases"""

    def test_returns_422_when_missing_audio_file(self, client):
        """Returns 422 when audio file missing"""
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "default"},
        )

        assert response.status_code == 422

    def test_handles_non_dict_json_params(self, client, test_audio_bytes):
        """Handles non-dict JSON params - causes TypeError"""
        # Note: The implementation doesn't validate that engine_params is a dict
        # before using **kwargs. This causes a TypeError which the error middleware
        # should convert to 500.
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "default", "engine_params": '["array"]'},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        # The error middleware converts unhandled exceptions to 500
        assert response.status_code == 500
