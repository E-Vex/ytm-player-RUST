# Releasing ytm-player

The release flow is **tag-driven**. Pushing a `vX.Y.Z` tag triggers PyPI publish, which in turn triggers AUR publish.

## Quick checklist

1. **Bump version** — `src/ytm_player/__init__.py` → `__version__ = "X.Y.Z"` **and** `native/Cargo.toml` →
   `version = "X.Y.Z"`, then refresh the lock from `native/` with `cargo update --workspace --quiet`
   (a changed crate version must reach `Cargo.lock`; `maturin build --locked` in the Publish workflow
   fails otherwise). `__version__` is the single source of truth the tag guard checks and the AUR
   workflow derives `pkgver` from — do NOT hand-edit `pkgver=` in `aur/PKGBUILD`, CI overwrites it.
   A packaging test (`tests/test_packaging/test_native_wheel_config.py`) fails CI if the two
   versions drift apart, because maturin stamps the binary wheels from the Cargo version.
2. **Fold CHANGELOG** — collapse all `### Unreleased` blocks into one `### vX.Y.Z (YYYY-MM-DD)` entry
3. **Lint + test**:
   ```bash
   .venv/bin/ruff format src/ tests/
   .venv/bin/ruff check src/ tests/
   .venv/bin/pytest -x -q
   ```
4. **Commit + tag + push** — stage the release files explicitly and review the complete staged diff before committing:
   ```bash
   git add src/ytm_player/__init__.py CHANGELOG.md
   git diff --cached
   git commit -m "chore(release): vX.Y.Z"
   git tag vX.Y.Z
   git push origin master --tags
   ```
5. **Watch the workflows**:
   ```bash
   gh run watch --exit-status
   ```

## What automation handles

### `Publish` workflow (`.github/workflows/publish.yml`)

Triggered on tag push. Builds the full artifact set on a per-OS matrix, smoke-tests every wheel
against `ytm --version` in a fresh venv, uploads to PyPI via OIDC trusted publishing, creates a
GitHub Release with the matching CHANGELOG section and all artifacts attached.

- **Binary legs (maturin):** one `cp310-abi3` wheel per platform carrying the Rust extension —
  `manylinux2014/manylinux_2_17` for x86_64 and aarch64 (built on ubuntu through zig against the
  glibc 2.17 baseline; aarch64 is cross-compiled, which is why it has no install smoke test), a
  macOS `universal2` wheel (`MACOSX_DEPLOYMENT_TARGET=11.0`) and a Windows `win_amd64` wheel.
  Every leg verifies its wheel with `.github/scripts/check_native_wheel.py` — file name tag,
  WHEEL `Tag:` headers, `Root-Is-Purelib`, extension placement and the binary's machine type —
  so a mis-tagged or wrong-architecture wheel fails the release instead of reaching PyPI.
- **Pure leg (hatchling):** the sdist and the `py3-none-any` fallback wheel, exactly as before.
  The tag-matches-`__version__` guard runs here; a failure blocks every upload because the
  publish jobs need the whole matrix to succeed.
- The `wheels` job in `ci.yml` builds the same binary matrix (same pinned maturin/ziglang) on
  every push, so wheel-config regressions surface long before a release tag.

No API tokens — auth is via OIDC trusted-publisher entries at PyPI and TestPyPI.

### `Publish AUR` workflow (`.github/workflows/aur-publish.yml`)

Triggered after `Publish` succeeds. Checks out the default branch, derives the version from the triggering tag (falling back to `__version__` for manual dispatch), clones `ssh://aur@aur.archlinux.org/ytm-player-git.git` via SSH key, copies `PKGBUILD`, rewrites its `pkgver=` line to the real version (the committed `pkgver=` is only a placeholder), regenerates `.SRCINFO` via `scripts/regenerate_srcinfo.py` (pure Python — no Arch dependency on Ubuntu CI runners), commits and pushes.

The published PKGBUILD compiles the Rust extension from source: `cargo` is a makedepend, `build()` runs `native/build.sh`, and `package()` installs the compiled module beside the wheel's files (hatchling honours `.gitignore`, which excludes `*.so`, so the module cannot ride in the wheel). `arch` is `('x86_64' 'aarch64')`. The `source=` URL must point at the repository that carries `native/` (this Rust-transition repo) — a packaging test (`tests/test_packaging/test_aur_pkgbuild_native.py`) fails CI if it is pointed at a repo without the Rust code.

The `regenerate_srcinfo.py` script handles shell variable expansion (`${url}`, `$pkgname`) so PKGBUILD constructs like `source=("git+${url}.git")` resolve correctly in the emitted `.SRCINFO`.

## Local builds

PyPI and GitHub Release artifacts are built only by `publish.yml` from a clean checkout. Never upload a locally built `dist/`.

hatchling packages from the working tree and ignores `.git/info/exclude`. The sdist is an explicit allowlist (`[tool.hatch.build.targets.sdist]` in `pyproject.toml`) and `*.local.md` is in `.gitignore`, which keeps the known private-file locations out of a local build. Any other untracked file inside the allowlisted directories would still be packaged: `pytest tests/test_packaging` reports such files before anything is archived. Review them; stage intended project files explicitly, or exclude private files.

## Manual fallback (if AUR action fails)

```bash
git clone ssh://aur@aur.archlinux.org/ytm-player-git.git /tmp/ytm-player-aur
cp aur/PKGBUILD /tmp/ytm-player-aur/
# The committed pkgver= is a placeholder — set the real version, like CI does:
sed -i "s/^pkgver=.*/pkgver=X.Y.Z/" /tmp/ytm-player-aur/PKGBUILD
python3 scripts/regenerate_srcinfo.py /tmp/ytm-player-aur/PKGBUILD /tmp/ytm-player-aur/.SRCINFO
cd /tmp/ytm-player-aur
git add PKGBUILD .SRCINFO
git commit -m "Update to vX.Y.Z"
git push
rm -rf /tmp/ytm-player-aur
```

## First-time setup (one-time, already done — for reference)

### 1. AUR SSH key

Generate a dedicated key for GitHub Actions (separate from your personal AUR key):

```bash
ssh-keygen -t ed25519 -C "github-actions-aur-publish" -f ~/.ssh/aur_publish_key -N ""
```

Add the **public** key (`~/.ssh/aur_publish_key.pub`) to your AUR account at https://aur.archlinux.org/account/. AUR supports multiple SSH public keys per account — add it alongside your existing one, do not replace.

### 2. GitHub repo secrets

```bash
gh secret set AUR_SSH_PRIVATE_KEY --repo peternaame-boop/ytm-player < ~/.ssh/aur_publish_key
ssh-keyscan -t ed25519 aur.archlinux.org 2>/dev/null | gh secret set AUR_KNOWN_HOSTS --repo peternaame-boop/ytm-player
```

Verify both exist:

```bash
gh secret list --repo peternaame-boop/ytm-player | grep AUR
```

### 3. PyPI trusted publishing

Both PyPI and TestPyPI have a trusted-publisher entry pointing at this repo's `publish.yml`. No tokens stored anywhere — auth is via OIDC at workflow runtime.

| Field | Value |
|-------|-------|
| Owner | `peternaame-boop` |
| Repository | `ytm-player` |
| Workflow | `publish.yml` |
| Environment | `pypi` (or `testpypi` on TestPyPI) |

Configured at https://pypi.org/manage/account/publishing/ and https://test.pypi.org/manage/account/publishing/.

## Dry-run via TestPyPI

Trigger `Publish` manually from Actions → **Run workflow** → target `testpypi`. Builds + uploads to https://test.pypi.org/project/ytm-player/ without touching production.

The AUR workflow does NOT trigger on TestPyPI dry-runs — it gates on the Publish workflow's `event=push` condition (i.e., real tag pushes only).

## Distribution channels

| Channel | Maintained by | Update mechanism |
|---------|---------------|------------------|
| PyPI (native wheels: manylinux x86_64/aarch64, macOS universal2, Windows amd64 + pure fallback) | Us | Automated via `publish.yml` on tag push |
| AUR (`ytm-player-git`) | Us | Automated via `aur-publish.yml` after PyPI succeeds; the package builds the Rust extension from source (cargo makedepend) |
| GitHub Release | Us | Automated via `publish.yml` on tag push |
| NixOS (`flake.nix`) | Us | Manual — bump in repo on release; the flake compiles the Rust extension itself (`ytm-player-native` `buildRustPackage`, deps vendored from `native/Cargo.lock`) |
| Gentoo (GURU overlay) | @dsafxP (community) | Out of our hands — community maintainer updates the ebuild on their schedule |

## After the release

- Verify PyPI: https://pypi.org/project/ytm-player/
- Verify AUR: `curl -s https://aur.archlinux.org/cgit/aur.git/plain/PKGBUILD?h=ytm-player-git | grep '^pkgver='`
- Verify GitHub Release: `gh release view vX.Y.Z`
- (Optional) Mention `@dsafxP` in the GitHub Release body so they know there's a new version to package for Gentoo.
