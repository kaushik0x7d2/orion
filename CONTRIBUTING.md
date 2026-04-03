# Contributing

Thank you for your interest in contributing to Orion!

## Getting Started

1. Fork the repository and clone your fork
2. Install dependencies: `pip install -e ".[dev]"`
3. Build the Go backend (see README for platform-specific commands)
4. Run the test suite: `pytest tests/ -v`

## Development Guidelines

- All new features must include tests in `tests/`
- Security-sensitive code must include adversarial tests in `tests/test_adversarial.py`
- FHE-compatible neural network layers go in `orion/nn/`
- Core infrastructure (error handling, memory, crypto) goes in `orion/core/`
- Run the full test suite before submitting a PR: `pytest tests/ -v` (163 tests)

## Reporting Issues

Please open a GitHub issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python version, OS, and Go version

## Security Vulnerabilities

If you discover a security vulnerability, please report it responsibly by emailing the maintainers rather than opening a public issue.
