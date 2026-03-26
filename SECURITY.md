# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in `space-ml-sim`, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead:

1. **GitHub Security Advisories** (preferred): Go to the [Security tab](https://github.com/yaitsmesj/space-ml-sim/security/advisories) and click "Report a vulnerability"
2. **Email**: Contact the maintainers directly (see GitHub profile)

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix or mitigation**: Depends on severity, but we aim for:
  - Critical: 48 hours
  - High: 1 week
  - Medium: 2 weeks
  - Low: Next release

## Scope

This policy covers:
- The `space-ml-sim` Python package
- CI/CD configuration
- Dependencies with known vulnerabilities

Out of scope:
- Simulation results or scientific accuracy (these are not security issues)
- Feature requests

## Automated Scanning

This project runs automated security checks in CI on every PR:
- `pip-audit` for dependency vulnerabilities
- `bandit` for Python code security issues
- `detect-secrets` for accidentally committed credentials
