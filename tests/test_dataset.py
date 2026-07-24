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
            (class_dir / f"sample_{index}.png").write_bytes(
                content + f"\n# unique {index}".encode()
            )
    return source


def test_discover_images_maps_kaggle_class_names(tmp_path: Path) -> None:
    source = make_source_dataset(tmp_path)

    discovered = discover_images(source)

    assert {label: len(paths) for label, paths in discovered.items()} == {
        "defective": 10,
        "ok_front": 10,
    }


def test_discover_images_prefers_300_pixel_dataset_variant(tmp_path: Path) -> None:
    root = tmp_path / "download"
    fixture = (FIXTURES / "defective_sample.ppm").read_bytes()
    for variant, count in (("casting_data/casting_data", 1), ("casting_512x512", 2)):
        for class_name in ("def_front", "ok_front"):
            class_dir = root / variant / class_name
            class_dir.mkdir(parents=True)
            for index in range(count):
                (class_dir / f"sample_{index}.png").write_bytes(fixture)

    discovered = discover_images(root)

    assert {label: len(paths) for label, paths in discovered.items()} == {
        "defective": 1,
        "ok_front": 1,
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


def test_prepare_dataset_keeps_exact_duplicates_in_one_split(tmp_path: Path) -> None:
    source = make_source_dataset(tmp_path)
    duplicate_content = (FIXTURES / "defective_sample.ppm").read_bytes()
    for class_name in ("def_front", "ok_front"):
        class_dir = source / "train" / class_name
        (class_dir / "duplicate_a.png").write_bytes(duplicate_content)
        (class_dir / "duplicate_b.png").write_bytes(duplicate_content)

    output = tmp_path / "prepared"
    prepare_dataset(source, output)

    for label in ("defective", "ok_front"):
        duplicate_splits = {
            path.parts[-3]
            for path in output.glob(f"*/{label}/*")
            if path.read_bytes() == duplicate_content
        }
        assert len(duplicate_splits) == 1
