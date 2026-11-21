{
  description = "ytm-player: A full-featured YouTube Music TUI client for the terminal";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        # Pinned to a stable middle of the supported range (3.10..3.14).
        # Bump along with nixpkgs releases.
        python = pkgs.python313;

        # spotifyscraper is not in nixpkgs — build from PyPI.
        spotifyscraper = python.pkgs.buildPythonPackage rec {
          pname = "spotifyscraper";
          version = "3.9.2";
          # PEP 517 build (hatchling). The sdist also ships a Makefile whose
          # `install` target runs `pip install --upgrade pip`; with
          # pyproject = false the Python hooks don't engage and stdenv falls
          # back to `make install`, which fails in the hermetic sandbox (no
          # pip, no network) — that was issue #93.
          pyproject = true;

          src = python.pkgs.fetchPypi {
            inherit pname version;
            hash = "sha256-6ozyx0VjVAbaqBnlUEZ5zUoRGPjS9U/qwnQH0DNPs60=";
          };

          build-system = with python.pkgs; [ hatchling ];

          # 3.x's only hard dependency; the CLI/browser/media/mcp extras are
          # optional and ytm-player uses none of them.
          dependencies = with python.pkgs; [ httpx ];

          # Tests require network access.
          doCheck = false;

          pythonImportsCheck = [ "spotify_scraper" ];

          meta = {
            description = "Scrape Spotify tracks, albums, playlists and artist data";
            homepage = "https://github.com/AliAkhtari9/SpotifyScraper";
            license = pkgs.lib.licenses.mit;
          };
        };

        # Shared package version, parsed from the single source of truth
        # (src/ytm_player/__init__.py). The Rust crate's version in
        # native/Cargo.toml is pinned equal to it by a packaging test
        # (tests/test_packaging/test_native_wheel_config.py), so both
        # derivations can use this one value.
        version = (builtins.head (
          builtins.match ''.*__version__[[:space:]]*=[[:space:]]*"([^"]+)".*'' (
            builtins.readFile ./src/ytm_player/__init__.py
          )
        ));

        # The Rust/PyO3 extension, compiled separately from the Python
        # package. buildRustPackage vendors the dependencies from
        # native/Cargo.lock (importCargoLock — no cargoHash to maintain,
        # and the lock has no git dependencies), builds the cdylib and its
        # default install phase puts libytm_player_native.{so,dylib} in
        # $out/lib (cargoInstallHook copies the release-dir shared
        # libraries). pyo3's build script needs the interpreter (and its
        # headers) at build time to generate its config — any Python
        # >= 3.10 works, the abi3-py310 feature pins the limited-API
        # floor, so the binary loads on the interpreter the app runs on.
        ytm-player-native = pkgs.rustPlatform.buildRustPackage {
          pname = "ytm-player-native";
          inherit version;

          src = ./native;

          cargoLock.lockFile = ./native/Cargo.lock;

          # No test targets in the crate; skip cargo test entirely.
          doCheck = false;

          nativeBuildInputs = [ python ];

          # Same forward-compat escape hatch native/build.sh exports: lets
          # the build keep working if nixpkgs' python moves past the newest
          # interpreter this pyo3 release knows.
          PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1";
        };

        ytm-player = python.pkgs.buildPythonApplication {
          pname = "ytm-player";
          inherit version;

          pyproject = true;
          src = ./.;

          build-system = [ python.pkgs.hatchling ];

          # Install the compiled Rust/PyO3 extension beside the pure-Python
          # tree the wheel installs. It cannot ride along in the wheel:
          # hatchling honours .gitignore (which excludes *.so / *.pyd as
          # build artifacts) even in the git-less flake source, so the
          # module is placed after the wheel install instead — at the exact
          # import path (ytm_player/utils/_ytm_player_native.abi3.so) the
          # package's `from .utils._ytm_player_native import ...` sites and
          # native/build.sh use. Artifact name per platform matches
          # cargo's cdylib naming (see native/build.sh).
          postInstall =
            let
              # cargo's cdylib artifact name per platform
              libext = if pkgs.stdenv.hostPlatform.isDarwin then "dylib" else "so";
            in
            ''
              install -Dm755 ${ytm-player-native}/lib/libytm_player_native.${libext} \
                $out/${python.sitePackages}/ytm_player/utils/_ytm_player_native.abi3.so
            '';

          # Relax upper bounds that may conflict with nixpkgs versions.
          # nixpkgs may ship textual 9.x while pyproject.toml caps at <9.0.
          pythonRelaxDeps = [ "textual" ];

          # python-mpv on PyPI registers as "mpv" in dist-info.
          # Remove the PyPI name so pythonRuntimeDepsCheck doesn't fail.
          pythonRemoveDeps = [ "python-mpv" ];

          dependencies =
            (with python.pkgs; [
              textual
              ytmusicapi
              yt-dlp
              mpv            # provides python-mpv (dist-info Name: mpv)
              aiosqlite
              click
              pillow         # album art (moved from optional to core in v1.3.1)
              packaging
              pycryptodomex  # native AES for yt-dlp's browser cookie decryption in `ytm setup`
            ])
            # dbus-fast powers Linux MPRIS (playerctl / media keys / now-playing).
            # Core dep on Linux so MPRIS works out of the box; Linux-only because
            # it can't import on darwin (socket.CMSG_LEN), matching the pyproject
            # sys_platform marker.
            ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [ python.pkgs.dbus-fast ];

          optional-dependencies = with python.pkgs; {
            mpris = [ ];  # dbus-fast moved to core deps (Linux-only); kept for compat
            images = [ ];  # Pillow moved to core deps; kept for compat
            discord = [ pypresence ];
            lastfm = [ pylast ];
            transliteration = [ anyascii ];
            spotify = [
              spotipy
              spotifyscraper
              thefuzz
            ];
          };

          # Wrap the ytm binary so mpv and yt-dlp are on PATH.
          # python-mpv's ctypes path to libmpv.so is already patched by nixpkgs,
          # but mpv the CLI tool is still needed for some operations, and yt-dlp
          # must be findable on PATH even though we also import it as a library.
          makeWrapperArgs = [
            "--prefix"
            "PATH"
            ":"
            (pkgs.lib.makeBinPath [
              pkgs.mpv
              python.pkgs.yt-dlp
            ])
          ];

          # Tests require network access and mpv runtime.
          doCheck = false;

          pythonImportsCheck = [
            "ytm_player"
            "ytm_player.cli"
            "ytm_player.services.update_check"
            # The compiled extension must actually load from the installed
            # tree — this is what proves the postInstall copy above works.
            "ytm_player.utils._ytm_player_native"
          ];

          meta = {
            description = "A full-featured YouTube Music TUI client for the terminal";
            homepage = "https://github.com/peternaame-boop/ytm-player";
            license = pkgs.lib.licenses.mit;
            maintainers = [ ];
            mainProgram = "ytm";
            platforms = pkgs.lib.platforms.linux ++ pkgs.lib.platforms.darwin;
          };
        };
      in
      {
        packages = {
          default = ytm-player;
          ytm-player = ytm-player;

          # Variant with all optional features enabled.
          ytm-player-full = ytm-player.overridePythonAttrs (old: {
            dependencies =
              old.dependencies
              ++ old.optional-dependencies.mpris
              ++ old.optional-dependencies.discord
              ++ old.optional-dependencies.lastfm
              ++ old.optional-dependencies.transliteration
              ++ old.optional-dependencies.spotify;
          });
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ ytm-player ];

          packages =
            (with python.pkgs; [
              # Dev tools
              pytest
              pytest-asyncio
              pytest-cov
              ruff

              # Include all optional deps in the dev shell
              pillow
              pypresence
              pylast
              spotipy
              spotifyscraper
              thefuzz
              anyascii
            ])
            # Rust toolchain so `native/build.sh` (extension build) and
            # `cargo fmt` / `cargo clippy` work straight out of the dev
            # shell — mirroring what the CI native job runs.
            ++ (with pkgs; [
              cargo
              rustc
              rustfmt
              clippy
            ])
            # dbus-fast is Linux-only (socket.CMSG_LEN); guard it so `nix develop`
            # still works on darwin.
            ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [ python.pkgs.dbus-fast ]
            ++ [
              pkgs.mpv
            ];

          shellHook = ''
            echo "ytm-player dev shell"
            echo "  python: ${python.version}"
            echo "  run:    ytm"
            echo "  test:   pytest"
            echo "  lint:   ruff check src/ tests/"
            echo "  native: native/build.sh"
          '';
        };
      }
    )
    // {
      # Overlay for use in NixOS configurations:
      #   nixpkgs.overlays = [ ytm-player.overlays.default ];
      #   environment.systemPackages = [ pkgs.ytm-player ];
      overlays.default = final: prev: {
        ytm-player = self.packages.${prev.stdenv.hostPlatform.system}.default;
      };
    };
}
