"""Type-checking stub for the compiled PyO3 extension.

The real module is ``_ytm_player_native.abi3.so``, built by
``native/build.sh`` — a checkout without that build (the CI typecheck
job, source distributions) has nothing for the import to resolve
against, so pyright needs a static spelling of the module here. The
module-level ``__getattr__`` (PEP 484) leaves every engine export
``Any`` instead of enumerating the whole native API surface; the
runtime import still binds the real extension when it is built, and
the pure-Python fallbacks carry the package when it is not.
"""

from typing import Any

def __getattr__(name: str) -> Any: ...
