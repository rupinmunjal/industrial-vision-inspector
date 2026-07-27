import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/run_app.py"


def load_run_app_module():
    specification = importlib.util.spec_from_file_location("run_app_script", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_default_paths_use_repository_when_running_from_source() -> None:
    module = load_run_app_module()

    model_path, database_path = module.default_paths()

    assert model_path == SCRIPT.parents[1] / "models/casting_yolov8n_cls.pt"
    assert database_path == SCRIPT.parents[1] / "data/inspections.db"


def test_default_paths_separate_frozen_resources_and_writable_data(
    monkeypatch,
) -> None:
    module = load_run_app_module()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", "/bundle/resources", raising=False)
    monkeypatch.setattr(sys, "executable", "/bundle/industrial-vision-inspector")

    model_path, database_path = module.default_paths()

    assert model_path == Path("/bundle/resources/models/casting_yolov8n_cls.pt")
    assert database_path == Path("/bundle/data/inspections.db")
