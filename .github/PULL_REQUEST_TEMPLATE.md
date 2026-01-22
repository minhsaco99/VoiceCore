## Description
<!-- Provide a brief description of the changes. If this is a breaking change, explain why. -->

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] New Engine (STT/TTS provider)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Refactor (non-breaking code cleanup or optimization)
- [ ] Documentation update
- [ ] Performance improvement

## Checklist
- [ ] I have read the [CONTRIBUTING](CONTRIBUTING.md) guide
- [ ] My code follows the project's code style (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Tests pass (`make test`)
- [ ] Documentation updated (if needed)
- [ ] No sensitive information (API keys, secrets) included

## Related Issues
<!-- Use "Closes #123" to automatically link to an issue -->
Closes #

## Testing & Verification

### Automated Tests
- [ ] Unit tests added/updated
- [ ] All existing tests pass

### Manual Verification (if applicable)
<!-- Describe how you manually tested these changes -->

### API Endpoints Tested (if applicable)
- [ ] Batch endpoint (`POST /api/v1/stt/transcribe` or `/tts/synthesize`)
- [ ] SSE streaming (`POST .../stream`)
- [ ] WebSocket (`WS .../ws`)

### Engine-Specific Tests (if applicable)
<!-- If this PR affects STT/TTS engines -->
- **Engine type**: [STT / TTS]
- **Provider**: [e.g., Whisper, Azure, Google]
- **Model**: [e.g., large-v3, base]
- **Device tested**: [cuda / cpu]

## Security Impact
- [ ] No security implications
- [ ] Security impact (please describe below)
