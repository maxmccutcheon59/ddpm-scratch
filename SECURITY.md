# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please email **MaxMcCutcheon1@outlook.com** with:

- A description of the issue and impact
- Steps to reproduce (PoC if available)
- Your preferred contact for follow-up

Do **not** open a public GitHub issue for security-sensitive reports until a fix or disclosure plan is agreed.

We aim to acknowledge reports within 7 days.

## Scope notes

This repository is offline research / ML training code. It does not run a network service by default. Still in scope: dependency vulnerabilities, secret leakage in the repo, unsafe deserialization of untrusted checkpoints, and CI supply-chain issues.

## Incident basics

If a secret is ever committed: rotate the credential first, then purge history. Enable GitHub secret scanning / push protection on the repository when available.
