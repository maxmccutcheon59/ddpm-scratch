# Compliance Notes — ddpm-scratch

**Status:** Research / educational ML code. Not a consumer product, SaaS, or data-collecting service.

## Data inventory

| Data | Collected by this repo? | Storage | Notes |
|------|-------------------------|---------|-------|
| End-user PII | **No** | — | No accounts, analytics, or telemetry |
| MNIST / Fashion-MNIST images | Downloaded on demand via `torchvision` | Local `./data/` (gitignored) | Public research datasets; see licenses below |
| Model checkpoints | Produced locally by `scripts/train.py` | Local `./checkpoints/` (gitignored) | Do not commit |

## Dataset license notes

- **MNIST**: Originally released by Yann LeCun et al. for research; commonly treated as freely usable for research/education. Obtained here through `torchvision.datasets.MNIST`.
- **Fashion-MNIST**: Zalando research dataset (MIT License for the Fashion-MNIST project). Obtained through `torchvision.datasets.FashionMNIST`.

Users must verify current terms for their jurisdiction and use case. This project does not redistribute the datasets inside the git tree.

## Legal / regulatory checklist (flags)

| Item | Applies? | Action |
|------|----------|--------|
| Privacy Policy / ToS | No (no user data collection) | N/A |
| GDPR / US state privacy | No personal data processed | N/A — re-review if productized |
| COPPA / minors | No | N/A |
| Payments / PCI | No | N/A |
| HIPAA / health | No | N/A |
| AI disclosure | Research reimplementation | Cite Ho et al. 2020; do not claim paper results |
| Export controls | Standard open-source ML | Human review if redistributing under sanction regimes |
| IP / third-party code | MIT original code + PyTorch/torchvision deps | Keep LICENSE; do not vendor unlicensed assets |
| Biometric / face data | No (digits/clothes glyphs only) | N/A |

## Security controls in this repo

- `.gitignore` excludes `.env*`, keys, checkpoints, datasets
- CI: `pytest`, `pip-audit`, `gitleaks`
- `SECURITY.md` with contact MaxMcCutcheon1@outlook.com
- No network server, auth, or PII handling

## Items needing human / lawyer review before productization

- If wrapping this as a public generative API: Privacy Policy, ToS, abuse monitoring, and model-output disclosure.
- If training on non-public or personal images: lawful basis, retention, and dataset licensing review.

## Authorization

Security scanning in CI targets **this** repository only (owned by maxmccutcheon59).

## CI workflow placement

The OAuth token used to publish this repo has scopes `gist`, `read:org`, `repo` but **not** `workflow`.
GitHub rejects both `git push` and Contents API writes to `.github/workflows/*` without that scope (HTTP 404).
The workflow YAML therefore ships at `ci/github-workflows/ci.yml` with enable instructions in `ci/README.md`.
This is a tooling limitation, not an intentional weakening of CI controls.
