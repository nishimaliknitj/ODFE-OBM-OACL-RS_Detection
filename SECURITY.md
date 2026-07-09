# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ |
| < 1.0   | ❌ |

## Reporting a Vulnerability

This is a research codebase (training/evaluation pipelines for object detection
models). It is not intended for deployment as a network-facing service. That
said, if you discover:

- A vulnerability in a dependency pinned by `requirements.txt` / `environment.yml`
- An unsafe deserialization path (e.g., in checkpoint loading via `torch.load`)
- Any code path that executes untrusted input

please report it privately by emailing **rahul.malik@galgotiasuniversity.edu.in**
with a description and reproduction steps. Do not open a public issue for
security reports. We aim to acknowledge reports within 5 business days.

## Known Considerations

- `torch.load` is used for checkpoint loading. Only load checkpoints from
  trusted sources; untrusted `.pt`/`.pth` files can execute arbitrary code via
  pickle. We recommend `weights_only=True` where supported by your PyTorch version.
- Dataset download scripts (`scripts/download_*.sh`) fetch from official
  third-party dataset hosts; verify checksums where provided.
