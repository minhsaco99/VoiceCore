# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email the maintainer directly at: [minhsaco99@gmail.com](mailto:minhsaco99@gmail.com)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix timeline**: Depends on severity

## Project-Specific Security

- **Audio Data**: Audio files may contain sensitive information. Consider implementing data retention policies and avoid logging audio content.
- **WebSocket Connections**: Implement rate limiting and connection limits in production.
- **File Upload Limits**: The API accepts audio uploads (default 25MB max). Adjust `MAX_AUDIO_SIZE` based on your needs.
- **Model Security**: Only use models from trusted sources (e.g., Hugging Face official repositories).

## Authentication

This API does not include built-in authentication. In production:

- Deploy behind an API gateway with authentication
- Use network-level access controls
- Never expose directly to the public internet without protection

## Dependency Security

- Keep dependencies updated regularly
- Run `pip audit` to check for known vulnerabilities
