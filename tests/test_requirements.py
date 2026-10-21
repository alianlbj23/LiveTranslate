from pathlib import Path

import pytest


import re

# The installer (LiveTranslate.exe runs it hidden) and the update window install packages
INSTALL_ENTRYPOINTS = ("launcher/setup.ps1", "launcher/update_window.ps1")


def _requirement_lines() -> set[str]:
    return {
        line.split("#", 1)[0].strip().lower()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }


def test_funasr_uses_published_dependency_metadata():
    """Install a current FunASR normally instead of copying its dependencies."""
    requirements = _requirement_lines()

    assert {
        "funasr>=1.3.28",
        "hydra-core>=1.3.2",
        "soundfile>=0.12.1",
    } <= requirements
    assert "editdistance-s>=1.0.0" not in requirements


def test_numpy_numba_versions_support_python_312_resolution():
    requirements = _requirement_lines()

    assert "numpy>=1.24.0,<2.3" in requirements
    assert "numba>=0.59.0" in requirements


@pytest.mark.parametrize("path", INSTALL_ENTRYPOINTS)
def test_install_entrypoint_resolves_funasr_dependencies(path: str):
    installer = Path(path).read_text(encoding="utf-8-sig").lower()

    assert re.search(r'"?-r"?[\s,(]+[^\n]*requirements\.txt', installer)
    assert "--no-deps" not in installer


def test_update_stops_when_dependency_installation_fails():
    """A failed pip step ends the update window's steps (no later step, no restart); the
    run itself is tested in test_app_update.py."""
    updater = Path("launcher/update_window.ps1").read_text(encoding="utf-8-sig")
    steps = updater[updater.index('"steps" {'):updater.index('"launch" {')]
    assert steps.index("if ($code -ne 0) { Set-Failed") < steps.index("$script:StepIndex++")


@pytest.mark.parametrize("path", INSTALL_ENTRYPOINTS)
def test_install_entrypoints_pin_yasbd_and_drop_pysbd(path: str):
    installer = Path(path).read_text(encoding="utf-8-sig").lower()

    assert 'yasbd-lib>=0.15,<1.0' in installer
    assert "install pysbd" not in installer


def test_readmes_do_not_describe_the_removed_editdistance_workaround():
    for path in (Path("README.md"), Path("README_zh.md")):
        text = path.read_text(encoding="utf-8").lower()
        assert "--no-deps" not in text
        assert "editdistance-s" not in text
