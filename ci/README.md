# CI workflow source

GitHub rejected pushes of `.github/workflows/*` from this environment because the
active OAuth token lacks the `workflow` scope (Contents API returns 404 for the
same path). The workflow definition therefore lives here:

- [`ci/github-workflows/ci.yml`](github-workflows/ci.yml) — pytest + pip-audit + gitleaks

## Enable Actions (one-time, repo admin with `workflow` scope)

```bash
mkdir -p .github/workflows
cp ci/github-workflows/ci.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Enable CI workflow"
git push
```

Or paste the file in the GitHub UI under **Actions → New workflow**.

Jobs: `test` (pytest + pip-audit on Python 3.10/3.12) and `gitleaks`.
