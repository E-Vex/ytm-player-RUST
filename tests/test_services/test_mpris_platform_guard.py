"""Import-safety guards for the MPRIS module (#106, #113).

dbus-fast is Linux-only — on Windows it raises AttributeError at import
(socket.CMSG_LEN). Because app/_app.py imports MPRISService unconditionally,
importing ytm_player.services.mpris must never pull in dbus_fast off Linux,
regardless of whether dbus-fast is present in the environment.

On Linux, a broken dbus-fast build can also raise TypeError while the MPRIS
interface classes are being *defined* (dbus-fast compiled with Cython 3.2.6
under Python 3.14 — #113). The module must degrade to DBUS_AVAILABLE=False
instead of failing to import.

These tests force sys.platform (and, for #113, a fake dbus_fast) in a fresh
subprocess interpreter so the module's guards run as they would on the target
system, without poisoning this test session's already-imported module state.
"""

import subprocess
import sys
import textwrap


def _run_import_under_platform(platform: str) -> str:
    """Import the mpris module with sys.platform forced to *platform*.

    Returns the subprocess stdout ("OK" on success); raises on assertion
    failure inside the child so the test reports the child's traceback.
    """
    script = textwrap.dedent(
        f"""
        import sys
        sys.platform = {platform!r}

        import ytm_player.services.mpris as mpris

        # The platform gate must keep dbus unavailable...
        assert mpris._DBUS_AVAILABLE is False, "_DBUS_AVAILABLE should be False"
        # ...and must never have imported the Linux-only library.
        assert "dbus_fast" not in sys.modules, "dbus_fast was imported off-Linux"
        # The service class must still be importable (app imports it directly).
        assert mpris.MPRISService is not None
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, (
        f"child failed (platform={platform}):\n{result.stdout}\n{result.stderr}"
    )
    return result.stdout.strip()


def test_mpris_import_safe_on_windows():
    assert _run_import_under_platform("win32") == "OK"


def test_mpris_import_safe_on_macos():
    assert _run_import_under_platform("darwin") == "OK"


def test_mpris_import_degrades_on_broken_dbus_fast_build():
    """A dbus-fast build that raises TypeError at interface-definition time
    (Cython 3.2.6 miscompile under Python 3.14 — #113) must degrade to
    DBUS_AVAILABLE=False, not break the module import.

    The fake dbus_fast imports cleanly but its ``method()`` decorator raises
    TypeError when applied — mirroring the real failure, where the crash
    happens in *our* class bodies, past the import-time guard.
    """
    script = textwrap.dedent(
        """
        import logging
        import sys
        import types

        # Explicit stderr handler so the assertion below doesn't depend on
        # logging's implicit last-resort handler.
        logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

        fake = types.ModuleType("dbus_fast")
        fake.Variant = lambda *a, **k: None

        aio = types.ModuleType("dbus_fast.aio")
        aio.MessageBus = type("MessageBus", (), {})

        constants = types.ModuleType("dbus_fast.constants")
        constants.PropertyAccess = type("PropertyAccess", (), {"READ": "read"})

        service = types.ModuleType("dbus_fast.service")
        service.ServiceInterface = type(
            "ServiceInterface", (), {"__init__": lambda self, name: None}
        )
        service.dbus_property = lambda access=None: (lambda fn: fn)
        service.signal = lambda: (lambda fn: fn)

        def _broken_method():
            def deco(fn):
                raise TypeError(
                    "__annotate__ called with unexpected argument (simulated "
                    "Cython 3.2.6 cyfunction failure)"
                )

            return deco

        service.method = _broken_method

        fake.aio = aio
        fake.constants = constants
        fake.service = service
        sys.modules["dbus_fast"] = fake
        sys.modules["dbus_fast.aio"] = aio
        sys.modules["dbus_fast.constants"] = constants
        sys.modules["dbus_fast.service"] = service

        sys.platform = "linux"

        import ytm_player.services.mpris as mpris

        assert mpris.DBUS_AVAILABLE is False, "broken build should degrade"
        assert mpris.MPRISService is not None
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, f"child failed:\n{result.stdout}\n{result.stderr}"
    assert result.stdout.strip() == "OK"
    # The broken-build path must be loud: logger.exception → stderr via the
    # last-resort handler in this bare interpreter (→ ytm.log in the app).
    assert "MPRIS D-Bus interfaces unavailable" in result.stderr
