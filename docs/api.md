# API Documentation

Complete API reference for Voice Engine API.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently no authentication is required. For production deployments, configure authentication via middleware.

## Endpoints

### Health & Discovery

#### GET /health

Basic health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

#### GET /ready

Readiness check - verifies all configured engines are initialized and ready.

**Response:**

```json
{
  "ready": true,
  "stt_engines": {
    "whisper": true
  },
  "tts_engines": {}
}
```

#### GET /engines

List all available engines with their status and capabilities.

**Response:**

```json
{
  "engines": [
    {
      "name": "whisper",
      "type": "stt",
      "ready": true,
      "engine_name": "faster-whisper",
      "supported_formats": ["wav", "mp3", "flac", "ogg", "m4a", "opus"]
    }
  ]
}
```

---

### Speech-to-Text (STT)

#### POST /stt/transcribe

Batch transcription - upload audio file and receive complete transcription.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `engine` | string | Yes | Engine name (e.g., "whisper") |
| `language` | string | No | Language hint (e.g., "en", "es") |
| `engine_params` | string | No | JSON string with engine-specific parameters |

**Request:**

- Content-Type: `multipart/form-data`
- Body: `file` - Audio file (wav, mp3, flac, ogg, m4a, opus)

**Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/stt/transcribe?engine=whisper&language=en" \
  -F "file=@audio.wav"
```

**Response:**

```json
{
  "text": "Hello, how are you today?",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 0.5,
      "text": "Hello,",
      "confidence": 0.95
    },
    {
      "start": 0.5,
      "end": 1.2,
      "text": "how",
      "confidence": 0.92
    }
  ],
  "performance_metrics": {
    "latency_ms": 1250.5,
    "processing_time_ms": 1200.0,
    "audio_duration_ms": 3500.0,
    "real_time_factor": 0.34
  }
}
```

#### POST /stt/transcribe/stream

SSE (Server-Sent Events) streaming transcription - receive progressive results as the audio is processed.

**Query Parameters:**

Same as `/stt/transcribe`.

**Request:**

Same as `/stt/transcribe`.

**Response:**

Server-Sent Events stream with two event types:

1. **chunk** - Partial transcription results

```
event: chunk
data: {"text": "Hello,", "timestamp": 0.0, "confidence": null, "chunk_latency_ms": 45.2}
```

2. **complete** - Final complete response

```
event: complete
data: {"text": "Hello, how are you today?", "language": "en", "segments": [...], "performance_metrics": {...}}
```

**Example:**

```bash
curl -N -X POST "http://localhost:8000/api/v1/stt/transcribe/stream?engine=whisper" \
  -F "file=@audio.wav"
```

**JavaScript Client Example:**

```javascript
const formData = new FormData();
formData.append('file', audioFile);

const eventSource = new EventSource(
  'http://localhost:8000/api/v1/stt/transcribe/stream?engine=whisper',
  { method: 'POST', body: formData }
);

eventSource.addEventListener('chunk', (event) => {
  const chunk = JSON.parse(event.data);
  console.log('Partial:', chunk.text);
});

eventSource.addEventListener('complete', (event) => {
  const result = JSON.parse(event.data);
  console.log('Final:', result.text);
  eventSource.close();
});
```

#### WS /stt/transcribe/ws

WebSocket endpoint for real-time transcription.

**Protocol:**

1. Client connects to WebSocket
2. Client sends config message (JSON):
   ```json
   {
     "engine": "whisper",
     "language": "en",
     "engine_params": {}
   }
   ```
3. Client sends audio chunks (binary data)
4. Client sends `"END"` (text) or disconnects when done
5. Server sends results:
   - `{"type": "chunk", "data": {...}}` - Partial results
   - `{"type": "complete", "data": {...}}` - Final result
   - `{"type": "error", "message": "..."}` - Error

**JavaScript Client Example:**

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/stt/transcribe/ws');

ws.onopen = () => {
  // Send config first
  ws.send(JSON.stringify({
    engine: 'whisper',
    language: 'en'
  }));

  // Then send audio chunks
  ws.send(audioChunk1);  // ArrayBuffer
  ws.send(audioChunk2);

  // Signal end of audio
  ws.send('END');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'chunk') {
    console.log('Partial:', message.data.text);
  } else if (message.type === 'complete') {
    console.log('Final:', message.data.text);
  } else if (message.type === 'error') {
    console.error('Error:', message.message);
  }
};
```

---

### Text-to-Speech (TTS)

#### POST /tts/synthesize

Batch synthesis - convert text to speech audio.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `engine` | string | Yes | Engine name (e.g., "coqui") |
| `text` | string | Yes | Text to synthesize |
| `voice` | string | No | Voice name/ID to use |
| `speed` | float | No | Speech speed multiplier (0 < speed <= 3.0, default: 1.0) |
| `engine_params` | string | No | JSON string with engine-specific parameters |

**Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/tts/synthesize?engine=coqui&text=Hello%20world&voice=en-US-1&speed=1.0"
```

**Response:**

```json
{
  "audio_data": "<base64-encoded-audio>",
  "sample_rate": 22050,
  "duration_seconds": 1.5,
  "format": "wav",
  "performance_metrics": {
    "latency_ms": 250.5,
    "processing_time_ms": 200.0,
    "audio_duration_ms": 1500.0,
    "real_time_factor": 0.13,
    "characters_processed": 11
  }
}
```

#### POST /tts/synthesize/stream

SSE (Server-Sent Events) streaming synthesis - receive progressive audio chunks.

**Query Parameters:**

Same as `/tts/synthesize`.

**Response:**

Server-Sent Events stream with two event types:

1. **chunk** - Partial audio data

```
event: chunk
data: {"audio_data": "<base64-chunk>", "sequence_number": 0, "chunk_latency_ms": 25.5}
```

2. **complete** - Final complete response

```
event: complete
data: {"audio_data": "<base64-full-audio>", "sample_rate": 22050, "duration_seconds": 1.5, "format": "wav", "performance_metrics": {...}}
```

**Example:**

```bash
curl -N -X POST "http://localhost:8000/api/v1/tts/synthesize/stream?engine=coqui&text=Hello%20world"
```

**JavaScript Client Example:**

```javascript
const params = new URLSearchParams({
  engine: 'coqui',
  text: 'Hello, how are you today?',
  voice: 'en-US-1',
  speed: '1.0'
});

const eventSource = new EventSource(
  `http://localhost:8000/api/v1/tts/synthesize/stream?${params}`
);

eventSource.addEventListener('chunk', (event) => {
  const chunk = JSON.parse(event.data);
  // Process audio chunk
  console.log('Chunk:', chunk.sequence_number);
});

eventSource.addEventListener('complete', (event) => {
  const result = JSON.parse(event.data);
  console.log('Complete, duration:', result.duration_seconds);
  eventSource.close();
});
```

---

## Data Models

### STTResponse

Complete transcription response.

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Complete transcribed text |
| `language` | string | Detected or specified language |
| `segments` | Segment[] | Word-level timestamps (optional) |
| `performance_metrics` | STTPerformanceMetrics | Performance metrics |

### Segment

Word or phrase with timing information.

| Field | Type | Description |
|-------|------|-------------|
| `start` | float | Start time in seconds |
| `end` | float | End time in seconds |
| `text` | string | The word or phrase |
| `confidence` | float | Confidence score (0.0 - 1.0) |

### STTChunk

Streaming chunk for partial results.

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Partial transcription text |
| `timestamp` | float | Position in audio (seconds) |
| `confidence` | float | Confidence score |
| `chunk_latency_ms` | float | Processing latency for this chunk |

### STTPerformanceMetrics

Performance metrics for transcription.

| Field | Type | Description |
|-------|------|-------------|
| `latency_ms` | float | Total end-to-end latency |
| `processing_time_ms` | float | Model processing time |
| `audio_duration_ms` | float | Audio duration |
| `real_time_factor` | float | Processing time / audio duration |
| `time_to_first_token_ms` | float | Time to first result (streaming) |
| `total_stream_duration_ms` | float | Total streaming duration |
| `total_chunks` | int | Number of chunks (streaming) |

### TTSResponse

Complete synthesis response.

| Field | Type | Description |
|-------|------|-------------|
| `audio_data` | bytes | Complete generated audio |
| `sample_rate` | int | Audio sample rate in Hz |
| `duration_seconds` | float | Audio duration in seconds |
| `format` | string | Audio format (wav, mp3, etc.) |
| `performance_metrics` | TTSPerformanceMetrics | Performance metrics |

### TTSChunk

Streaming chunk for partial audio.

| Field | Type | Description |
|-------|------|-------------|
| `audio_data` | bytes | Audio chunk bytes |
| `sequence_number` | int | Chunk sequence for ordering |
| `chunk_latency_ms` | float | Generation latency for this chunk |

### TTSPerformanceMetrics

Performance metrics for synthesis.

| Field | Type | Description |
|-------|------|-------------|
| `latency_ms` | float | Total end-to-end latency |
| `processing_time_ms` | float | Model processing time |
| `real_time_factor` | float | Processing time / audio duration |
| `characters_per_second` | float | Text-to-audio synthesis speed |
| `time_to_first_byte_ms` | float | Time to first audio byte (streaming) |
| `total_stream_duration_ms` | float | Total streaming duration |
| `total_chunks` | int | Number of chunks (streaming) |

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters or audio |
| 404 | Not Found - Engine not found |
| 413 | Payload Too Large - Audio exceeds size limit |
| 422 | Unprocessable Entity - Validation error |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Engine not ready |

### Common Errors

**Engine not found:**
```json
{
  "detail": "STT engine 'unknown' not found"
}
```

**Invalid audio:**
```json
{
  "detail": "Audio is empty"
}
```

**Engine not ready:**
```json
{
  "detail": "Engine has been closed"
}
```

---

## Engine-Specific Parameters

### Whisper (faster-whisper)

Pass these via `engine_params` query parameter as JSON string.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beam_size` | int | 5 | Beam search size |
| `temperature` | float | 0.0 | Sampling temperature (0.0 = greedy) |
| `vad_filter` | bool | false | Enable Voice Activity Detection |
| `condition_on_previous_text` | bool | true | Use previous text as context |
| `compression_ratio_threshold` | float | 2.4 | Compression ratio threshold |
| `log_prob_threshold` | float | -1.0 | Log probability threshold |
| `no_speech_threshold` | float | 0.6 | No speech probability threshold |

**Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/stt/transcribe?engine=whisper&engine_params={\"beam_size\":3,\"vad_filter\":true}" \
  -F "file=@audio.wav"
```

---

## Rate Limiting

No rate limiting is configured by default. For production deployments, configure rate limiting via middleware or API gateway.

## CORS

CORS is configured via the `CORS_ORIGINS` environment variable. Default allows all origins (`*`).
