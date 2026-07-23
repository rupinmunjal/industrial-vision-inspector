from pathlib import Path

from industrial_vision_inspector.dataset import discover_images, prepare_dataset

FIXTURES = Path(__file__).parent / "fixtures"


def make_source_dataset(root: Path) -> Path:
    source = root / "casting_data"
    for class_name, fixture in (
        ("def_front", "defective_sample.ppm"),
        ("ok_front", "ok_sample.ppm"),
    ):
        class_dir = source / "train" / class_name
        class_dir.mkdir(parents=True)
        content = (FIXTURES / fixture).read_bytes()
        for index in range(10):
            (class_dir / f"sample_{index}.png").write_bytes(content)
    return source


def test_discover_images_maps_kaggle_class_names(tmp_path: Path) -> None:
    source = make_source_dataset(tmp_path)

    discovered = discover_images(source)

    assert {label: len(paths) for label, paths in discovered.items()} == {
        "defective": 10,
        "ok_front": 10,
    }


def test_prepare_dataset_creates_deterministic_split_and_cache(tmp_path: Path) -> None:
    source = make_source_dataset(tmp_path)
    output = tmp_path / "prepared"

    counts = prepare_dataset(source, output)
    cached_counts = prepare_dataset(source, output)

    assert counts == {
        "train": {"defective": 7, "ok_front": 7},
        "val": {"defective": 1, "ok_front": 1},
        "test": {"defective": 2, "ok_front": 2},
    }
    assert cached_counts == counts
    assert (output / "split_manifest.json").is_file()
    assert len(list((output / "train" / "defective").glob("*.png"))) == 7

