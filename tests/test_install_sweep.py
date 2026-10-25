import runpy
from pathlib import Path

import pytest

INSTALLER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/install_sweep.py")
)
install_units = INSTALLER["install_units"]
systemd_path = INSTALLER["systemd_path"]


@pytest.fixture
def project(tmp_path):
    project = tmp_path / 'checkout with spaces % $ and "quotes"'
    source = project / "deploy/systemd"
    source.mkdir(parents=True)
    (source / "nexusrag-sweep.service.in").write_text(
        "[Service]\nType=oneshot\nWorkingDirectory=@PROJECT_DIR@\nExecStart=@UV_PATH@ run --no-sync python -m nexusrag.sweep_jobs --apply\n"
    )
    (source / "nexusrag-sweep.timer").write_text("[Timer]\nOnActiveSec=1min\n")
    return project


def test_generates_local_paths_and_links_timer(project, tmp_path):
    destination = tmp_path / "units"
    uv = tmp_path / "uv with $ and %"
    target = install_units(project, destination, uv)
    contents = target.read_text()
    assert f"WorkingDirectory={systemd_path(project)}" in contents
    assert f"ExecStart={systemd_path(uv, executable=True)} run" in contents
    assert "@PROJECT_DIR@" not in contents
    assert "@UV_PATH@" not in contents
    assert (
        destination / "nexusrag-sweep.timer"
    ).resolve() == project / "deploy/systemd/nexusrag-sweep.timer"
    assert not target.is_symlink()
    # Managed output is safe to regenerate after changing the uv location.
    install_units(project, destination, tmp_path / "different-uv")
    assert "different-uv" in target.read_text()


@pytest.mark.parametrize("existing", ["service", "timer"])
def test_preserves_unrelated_unit_files(project, tmp_path, existing):
    destination = tmp_path / "units"
    destination.mkdir()
    conflict = destination / f"nexusrag-sweep.{existing}"
    conflict.write_text("custom unit")
    with pytest.raises(ValueError, match="needs review"):
        install_units(project, destination, tmp_path / "uv")
    assert conflict.read_text() == "custom unit"
    assert len(list(destination.iterdir())) == 1


def test_migrates_old_service_symlink_without_writing_repo_source(project, tmp_path):
    source = project / "deploy/systemd/nexusrag-sweep.service"
    source.write_text("old source")
    destination = tmp_path / "units"
    destination.mkdir()
    target = destination / source.name
    target.symlink_to(source)
    install_units(project, destination, tmp_path / "uv")
    assert not target.is_symlink()
    assert source.read_text() == "old source"


def test_rejects_unrelated_symlink_before_writing(project, tmp_path):
    destination = tmp_path / "units"
    destination.mkdir()
    target = destination / "nexusrag-sweep.timer"
    target.symlink_to(tmp_path / "other-timer")
    with pytest.raises(ValueError, match="needs review"):
        install_units(project, destination, tmp_path / "uv")
    assert not (destination / "nexusrag-sweep.service").exists()


def test_path_escaping():
    assert systemd_path(Path('/a $b%/"c"'), executable=True) == '"/a $$b%%/\\"c\\""'
    with pytest.raises(ValueError, match="control characters"):
        systemd_path(Path("/path\nwith-newline"))
