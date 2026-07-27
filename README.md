# Industrial Vision Inspector

Industrial Vision Inspector is a desktop quality-control project for classifying top-view casting images as defective or acceptable. The target dataset contains submersible pump impeller images captured from one product type and camera angle. The project includes reproducible data preparation, OpenCV ingestion, a YOLOv8 classification wrapper, held-out evaluation, SQLite inspection history, a PySide6 operator interface, and CSV/PDF reporting.

Manual visual inspection can vary between operators and across a long shift. A small classification model can provide a consistent first-pass decision and preserve each result for review. This project is intentionally a demonstrator, not a claim of production-line readiness.

## Detection decision

The project uses a **classification-first** pipeline because the Kaggle dataset provides image-level `defective` and `ok_front` labels, not defect bounding boxes. YOLOv8 classification is faster and more reliable to train on modest hardware than a detector trained on noisy pseudo-boxes. Defective images will receive approximate OpenCV contour highlights for visual feedback; those highlights are not learned localization ground truth.

## Architecture

```mermaid
flowchart LR
    A[OpenCV ingestion] --> B[YOLOv8 classification]
    B --> C[Inspection result]
    C --> D[(SQLite)]
    C --> E[PySide6 UI]
    D --> E
    D --> F[CSV and PDF reports]
```

The ingestion, classification, storage, UI, and reporting components use one inspection path. Datasets, trained weights, captured webcam frames, inspection databases, and generated reports are intentionally excluded from Git; the compact evaluation metrics and confusion matrix are versioned for reproducibility.

## Setup

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The direct runtime, test, and packaging dependencies are pinned in `requirements.txt`. Ultralytics installs PyTorch as a dependency. On a CPU-only Linux machine, install the smaller CPU wheels before the requirements to avoid downloading unused CUDA libraries:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -r requirements.txt
```

### Kaggle credentials and dataset

1. Sign in to Kaggle and open **Settings**.
2. Under **API**, select **Create New Token** to download `kaggle.json`.
3. Move it to the Kaggle CLI location and restrict its permissions:

   ```bash
   mkdir -p ~/.kaggle
   mv /path/to/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. Download, cache, and prepare the dataset:

   ```bash
   python scripts/download_dataset.py
   ```

The archive is cached under `data/cache/`, extracted under `data/raw/`, and split under `data/processed/casting_cls/`. All three locations are ignored by Git. The preparation step selects the 7,348-image 300×300 dataset variant and uses a fixed seed for a 70/15/15 per-class train, validation, and test split. Byte-identical images are grouped into the same split to prevent exact-duplicate leakage. Re-running the script uses completion manifests instead of downloading or copying again.

To prepare an existing local copy without contacting Kaggle:

```bash
python scripts/download_dataset.py --source-dir /path/to/extracted/dataset
```

## Verify ingestion

Run the tests:

```bash
pytest
```

Create a contact sheet from the checked-in fixtures:

```bash
python scripts/verify_ingestion.py
```

The script writes `reports/ingestion_samples.jpg`. Pass a folder path to inspect other images, or add `--show` when a graphical display is available. The module accepts a video path or webcam index through `iter_frames`; webcam index `0` selects the default camera.

## Train and evaluate the classifier

After preparing the dataset, train the demo-scale model:

```bash
python scripts/train_model.py
```

The defaults use pretrained `yolov8n-cls` weights, 10 epochs, 224-pixel images, a batch size of 16, and seed 42. Pass `--device 0` for the first CUDA GPU or `--device cpu` to force CPU training. The best weights are copied to `models/casting_yolov8n_cls.pt`.

Evaluate those weights on the held-out test split:

```bash
python scripts/evaluate_model.py
```

This writes the following artifacts under `reports/metrics/`:

- `metrics.json` with accuracy, precision, recall, and F1
- `confusion_matrix.png`
- `prediction_samples.png` with correct and incorrect examples when both exist

Inspect one image through the same `Inspector.inspect(image)` path that the desktop UI will use:

```bash
python scripts/inspect_image.py /path/to/casting-image.jpeg
```

Defective classifications receive a rough OpenCV contour box. The box is visual feedback only and is not a learned defect location.

## Detection metrics

The model was trained on the deterministic, duplicate-grouped split with pretrained `yolov8n-cls` weights, 224-pixel inputs, batch size 16, and seed 42. CPU training stopped after epoch 9 because validation top-1 accuracy did not improve for three epochs. The best checkpoint reached 99.46% validation top-1 accuracy.

The following results come from the untouched 1,105-image test split. `defective` is the positive class.

| Metric | Result |
| --- | ---: |
| Accuracy | 99.46% |
| Precision | 100.00% |
| Recall | 99.05% |
| F1 | 99.52% |

The test set contained 472 true negatives, 627 true positives, no false positives, and 6 false negatives.

![Held-out confusion matrix](reports/metrics/confusion_matrix.png)

## Inspection storage

Inspection history is stored in a local SQLite database through a thin `sqlite3` data-access module. SQLite provides transactions and durable storage without requiring a database server, which fits a single-operator desktop application. The schema stores the inspection time, source image path, pass/fail result, model confidence, and optional operator notes. It does not store a defect subtype because the current classifier does not predict one.

Timestamps are normalized to UTC. History queries return newest records first and can filter by result or a half-open `[start, end)` date range. The project uses the Python standard library directly rather than adding an ORM for one table.

## Run the desktop application

Train the model first, then launch the operator interface from the repository root:

```bash
python scripts/run_app.py
```

The Inspect tab accepts one image, every supported image directly inside a folder, or a captured webcam frame. Folder work runs outside the GUI thread and writes each completed result to `data/inspections.db`. Webcam inference is operator-triggered rather than continuous; captured frames are saved under `data/captures/` so history rows refer to real files. The History tab shows the stored UTC timestamp, image path, result, confidence, and notes, with result and UTC date filters.

### Export inspection reports

Apply any result or UTC date filters in the History tab, then choose **Export CSV** or **Export PDF**. Both actions export the currently filtered records.

- CSV uses the standard library and includes ID, UTC timestamp, image path, result, confidence, and notes.
- PDF uses one ReportLab template with the inspection date range, pass/fail counts, defect rate, outcome chart, and up to four readable example images.
- Missing example image files are omitted from the PDF, but their inspection rows still contribute to the summary counts.

![Inspect view with a defective casting](docs/screenshots/inspect-view.png)

![SQLite-backed inspection history](docs/screenshots/history-view.png)

### UI smoke-test checklist

1. Launch the app and confirm the Inspect and History tabs open without an error.
2. Inspect known acceptable and defective held-out images; confirm the badge, confidence, and approximate defective highlight.
3. Select `tests/fixtures/` as a folder; confirm both images complete and two history rows appear.
4. Start the webcam, inspect one frame, and confirm the saved capture appears in history. This step requires local camera permission.
5. Filter history by result and UTC date, export both formats, and confirm each output opens correctly.
6. Restart the app and confirm the history rows persist.

For a headless startup check that loads the real weights and exits automatically:

```bash
QT_QPA_PLATFORM=offscreen python scripts/run_app.py \
  --database /tmp/industrial-vision-smoke.db \
  --smoke-test
```

## Package the desktop application

PyInstaller can produce a self-contained application folder for the current operating system. Train the model first so `models/casting_yolov8n_cls.pt` exists, then run:

```bash
pyinstaller --noconfirm --clean \
  --name industrial-vision-inspector \
  --windowed \
  --paths src \
  --add-data "models/casting_yolov8n_cls.pt:models" \
  scripts/run_app.py
```

Launch `dist/industrial-vision-inspector/industrial-vision-inspector`. The trained weights are bundled as a read-only application resource, while inspection history and webcam captures are written under the executable's `data/` directory. On Windows, use `;` instead of `:` in the `--add-data` value.

PyInstaller output is platform-specific, so build separately on each target operating system. The verified Linux application folder is approximately 1.4 GB because it contains PyTorch, Qt, OpenCV, and their native libraries; PySide6 6.11 Linux wheels also require glibc 2.34 or newer. This project intentionally does not include an installer, code signing, binary-size optimization, or a release pipeline.

## Current limitations

- Trained weights are not committed; reproduce them with `scripts/train_model.py` before running inference.
- Folder ingestion is intentionally non-recursive.
- Webcam behavior depends on local camera permissions and is not exercised in headless tests.
- The dataset supports one casting product and one camera viewpoint.
- OpenCV defect highlights are approximate and can select normal high-contrast geometry.
- Classification confidence is not calibrated. The six false negatives were predicted as acceptable with 84.9% to 99.2% confidence.
- Exact duplicate files are kept within one split, but the pipeline does not detect visually near-duplicate images.
