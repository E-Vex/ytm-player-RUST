<p align="center">
  <img src="https://raw.githubusercontent.com/peternaame-boop/ytm-player/master/docs/images/header.svg" alt="ytm-player — YouTube Music in your terminal — synced lyrics, vim keys, mpv backend. Runs on Linux, macOS, Windows. Free-tier supported." width="720" />
</p>

<p align="center">
  <a href="https://ytm-player.com"><img src="https://raw.githubusercontent.com/peternaame-boop/ytm-player/master/docs/images/website-button.svg" alt="Visit ytm-player.com" width="240" /></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/ytm-player/"><img src="https://img.shields.io/pypi/v/ytm-player?style=for-the-badge&logo=pypi&color=ff4e45&labelColor=0f0f0f&logoColor=ff4e45" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-ff4e45?style=for-the-badge&logo=python&labelColor=0f0f0f&logoColor=ff4e45" alt="Python 3.10+"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.98+-ff4e45?style=for-the-badge&logo=rust&labelColor=0f0f0f&logoColor=ff4e45" alt="Rust"></a>
  <a href="https://github.com/peternaame-boop/ytm-player/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/peternaame-boop/ytm-player/ci.yml?style=for-the-badge&logo=githubactions&labelColor=0f0f0f&logoColor=ff4e45" alt="CI"></a>
  <a href="https://github.com/peternaame-boop/ytm-player/blob/master/LICENSE"><img src="https://img.shields.io/github/license/peternaame-boop/ytm-player?style=for-the-badge&logo=opensourceinitiative&color=ff4e45&labelColor=0f0f0f&logoColor=ff4e45" alt="License"></a>
</p>

<p align="center">
  <a href="#install"><img src="https://img.shields.io/badge/Install-ff4e45?style=for-the-badge&labelColor=0f0f0f" alt="Install"></a>&nbsp;
  <a href="#quickstart"><img src="https://img.shields.io/badge/Quickstart-ff4e45?style=for-the-badge&labelColor=0f0f0f" alt="Quickstart"></a>&nbsp;
  <a href="#documentation"><img src="https://img.shields.io/badge/Documentation-ff4e45?style=for-the-badge&labelColor=0f0f0f" alt="Documentation"></a>&nbsp;
  <a href="https://github.com/peternaame-boop/ytm-player/blob/master/CONTRIBUTING.md"><img src="https://img.shields.io/badge/Contributing-ff4e45?style=for-the-badge&labelColor=0f0f0f" alt="Contributing"></a>&nbsp;
  <a href="https://github.com/peternaame-boop/ytm-player/blob/master/CHANGELOG.md"><img src="https://img.shields.io/badge/Changelog-ff4e45?style=for-the-badge&labelColor=0f0f0f" alt="Changelog"></a>
</p>

---

![ytm-player demo](https://raw.githubusercontent.com/peternaame-boop/ytm-player/master/docs/images/ytm-demo.gif)

> Available on **PyPI**, **AUR**, **NixOS**, and **Gentoo** — actively maintained with cross-platform support.

## Features

- **Vim-style keybindings** — j/k movement, multi-key sequences (`g s` for search, `g l` for library), count prefixes (`5j`)
- **Synced lyrics** — live-highlighted with LRCLIB fallback for tracks YouTube doesn't have, with title sanitization for better LRCLIB matches
- **mpv backend** — gapless audio, stream prefetching, broad codec support
- **Cross-platform native integrations** — MPRIS (Linux), Now Playing (macOS), media keys (Windows)
- **Theming** — 18+ Textual themes plus per-app color overrides in `theme.toml`
- **Spotify import** — pull playlists in via API or scraper fallback
- **CLI + IPC** — control a running TUI from another terminal (`ytm play`, `ytm pause`, etc.)
- **Free-tier support** — works without YouTube Music Premium
- **Session resume** — last-playing track + queue restored on launch
- **Local cache** — LRU audio cache for offline-like replay of previously heard tracks
- **Discord + Last.fm** — Rich Presence and scrobbling

## Requirements

- **Python 3.10+**
- **[mpv](https://mpv.io/)** (audio playback backend, must be installed system-wide)
- **[Rust toolchain](https://rustup.rs/) (1.98+)** — optional; only needed to build the native Rust extension. The pure-Python fallback is used automatically when the extension is not compiled.
- A YouTube Music account (free or Premium)

## Install

```bash
# PyPI (Linux / macOS / Windows)
pip install ytm-player

# Arch / CachyOS / Manjaro (AUR)
yay -S ytm-player-git
```

For NixOS, Gentoo, Windows-specific mpv DLL setup, source builds, and optional extras (Discord, Last.fm, Spotify import, etc.), see [docs/installation.md](https://github.com/peternaame-boop/ytm-player/blob/master/docs/installation.md).

## Native extension (Rust)

The formatting utilities (`ytm_player/utils/formatting.py`), BiDi text utilities (`ytm_player/utils/bidi.py`), transliteration (`ytm_player/utils/transliteration.py`), keymap sequence matching and key-name normalisation (`ytm_player/config/keymap.py` and `ytm_player/app/_keys.py`), chart-region normalisation (`ytm_player/services/regions.py`), yt-dlp option normalisation (`ytm_player/services/yt_dlp_options.py`), the single-instance argv heuristic (`ytm_player/ipc.py`), the diagnostics text pipeline (redaction, log-line filtering, and process matching in `ytm_player/utils/doctor.py` and `ytm_player/utils/logging.py`), the playback queue's ordering engine (`ytm_player/services/queue.py`), the track table's filter/sort view engine (`ytm_player/ui/widgets/track_table.py`), the session-state validation/coercion engine that guards the `session.json` restore path (`ytm_player/app/_session.py`), the Recently Played history-merge engine (occurrence stamping/identity, accepted-play supersession, pending-play parking/merging, and the All-tab local+account composition in `ytm_player/ui/pages/recently_played.py`), the navigation back/forward stack engine (the leaving-page snapshots, stack caps, same-page-back shortcut, forward-history invalidation and cached-state restore behind page navigation, plus the purge that drops nav state referencing deleted playlists, in `ytm_player/app/_navigation.py`), the Browse home-feed shelf classification engine (the event-vs-chart partition and priority pill ordering behind the Charts tab, plus the playlist-row extraction with per-id dedup behind the Playlists tab, in `ytm_player/ui/pages/browse.py`), the search-result label formatting engine (the item-to-label composer behind the Albums/Artists/Playlists panels of the search results area, in `ytm_player/ui/pages/search.py`), the sidebar pane-routing state machine (the per-page Playlists-sidebar visibility, the visible-pane list, the focused-widget pane resolution, the focus gate, and the Ctrl+w left/right/cycle movement decisions in `ytm_player/app/_sidebar.py`), the playlist panel view engine (the row label, footer count label, `/`-filter pass, playlist-id matching behind the optimistic sidebar mutations, and count arithmetic behind the sidebar's Playlists panel, in `ytm_player/ui/sidebars/playlist_sidebar.py`), the context-page view-model assembly (the album and playlist header compositions and the artist discography — top songs, albums+singles, related artists, and the see-all section selection behind the discography expansion, in `ytm_player/ui/pages/context.py`), the count/sequence key-input state machine (the count-prefix accumulation and sequence-buffer resolution behind every keystroke's dispatch, in `ytm_player/app/_keys.py`), the `_handle_action` action-dispatch engine (the dispatch-table walk behind every resolved action's branch, the pane decision behind the routed navigation actions, the CURRENT_CONTEXT album-target extraction and the TOGGLE_SHUFFLE lock gate, also in `ytm_player/app/_keys.py`), and the play-history row engines (the skip gate and insert-row assembly behind `HistoryManager.log_play`, plus the stored-vs-new listened-seconds decision behind `update_play_listened_seconds`, in `ytm_player/services/history.py`) are implemented in Rust via [PyO3](https://pyo3.rs/) for better performance. The pure-Python fallbacks are always used automatically when the compiled extension is unavailable, so nothing is required to run ytm-player — but building the native extension is recommended for source installs.

### Prerequisites

- [Rust toolchain](https://rustup.rs/) (1.98+)
- Python 3.10+ dev headers (`python3-dev`)

### Build

```bash
cd native
./build.sh
```

This compiles `native/src/lib.rs` into `_ytm_player_native.abi3.so` and copies it into `src/ytm_player/utils/`. Re-run `./build.sh` after any change to the Rust source.

The extension is built with PyO3's stable ABI (`abi3`), so it remains loadable across Python minor versions within the same 3.x series.

## Quickstart

```bash
ytm setup    # one-time auth (auto-detects browser cookies)
ytm          # launch the TUI
```

Windows: replace `ytm` with `py -m ytm_player`.

## Documentation

| Topic | Link |
|-------|------|
| Per-platform install + optional extras | [docs/installation.md](https://github.com/peternaame-boop/ytm-player/blob/master/docs/installation.md) |
| `config.toml` + `theme.toml` reference | [docs/configuration.md](https://github.com/peternaame-boop/ytm-player/blob/master/docs/configuration.md) |
| Full keyboard + mouse keybindings | [docs/keybindings.md](https://github.com/peternaame-boop/ytm-player/blob/master/docs/keybindings.md) |
| All `ytm` CLI subcommands | [docs/cli-reference.md](https://github.com/peternaame-boop/ytm-player/blob/master/docs/cli-reference.md) |
| Spotify playlist import | [docs/spotify-import.md](https://github.com/peternaame-boop/ytm-player/blob/master/docs/spotify-import.md) |
| Troubleshooting (mpv / auth / MPRIS / macOS / cache) | [docs/troubleshooting.md](https://github.com/peternaame-boop/ytm-player/blob/master/docs/troubleshooting.md) |
| File layout + stack | [docs/architecture.md](https://github.com/peternaame-boop/ytm-player/blob/master/docs/architecture.md) |
| Contributing | [CONTRIBUTING.md](https://github.com/peternaame-boop/ytm-player/blob/master/CONTRIBUTING.md) |
| Security policy | [SECURITY.md](https://github.com/peternaame-boop/ytm-player/blob/master/SECURITY.md) |

## Contributors

Thanks to [@dmnmsc](https://github.com/dmnmsc), [@Villoh](https://github.com/Villoh), [@valkyrieglasc](https://github.com/valkyrieglasc), [@dsafxP](https://github.com/dsafxP), [@Thayrov](https://github.com/Thayrov), [@glywil](https://github.com/glywil), [@Kineforce](https://github.com/Kineforce), [@CarterSnich](https://github.com/CarterSnich), [@Tohbuu](https://github.com/Tohbuu), [@nitsujri](https://github.com/nitsujri), [@uhs-robert](https://github.com/uhs-robert), [@moschi](https://github.com/moschi), [@firedev](https://github.com/firedev), [@wgordon17](https://github.com/wgordon17), [@gitiy1](https://github.com/gitiy1), [@hanandewa5](https://github.com/hanandewa5), [@aimar-a](https://github.com/aimar-a), [@Gimar250](https://github.com/Gimar250), [@Wiibleyde](https://github.com/Wiibleyde), [@szx19970521](https://github.com/szx19970521), [@aaguilar-hub](https://github.com/aaguilar-hub), [@TheoBassaw](https://github.com/TheoBassaw), [@holstvoogd](https://github.com/holstvoogd), and [@ProfessionalGriefer](https://github.com/ProfessionalGriefer) for bug reports, fixes, packaging, and platform support.

## Changelog

See [CHANGELOG.md](https://github.com/peternaame-boop/ytm-player/blob/master/CHANGELOG.md) for the full release history.
