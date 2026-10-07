# Foreign Object Detection for UAV Power-Line Inspection

This repository contains the training-free, normal-template-guided foreign-object localization method described in “Training-free foreign object detection for UAV power line inspection using feature-aligned multi-scale image comparison”.

The method compares a historical normal template with a current inspection image. It registers the image pair (IFM), evaluates complementary grid slices at several scales (MGS), and ranks local regions with HaarPSI (SAM). It does not train a foreign-object classifier.

## Repository contents

- src/foreign_object_detection/ifm.py: AKAZE matching, ratio filtering, RANSAC homography, and warping.
- src/foreign_object_detection/mgs.py: non-overlapping corresponding grids.
- src/foreign_object_detection/sam.py: configurable HaarPSI scoring and candidate filtering.
- src/foreign_object_detection/pipeline.py: the complete pair-level inference pipeline.
- src/foreign_object_detection/evaluation.py: strict one-to-one IoU matching and precision/recall.
- scripts/run_detection.py: directory-based inference entry point.
- scripts/evaluate.py: evaluation against local Pascal VOC-style XML files.
- configs/default.json: audited source-behavior profile.
- configs/manuscript_literal.json: profile following the manuscript's Hamming/grayscale description.

The proposed IFM–MGS–SAM method is training-free, so this repository has no training entry point for the proposed method. Comparison-model code, baseline training, model weights, figure-generation scripts, exploratory metrics, historical logs, images, and annotations are not included. The YOLACT training/video demo in the source directory is a separate cable-and-tower instance-segmentation project and is not part of this manuscript's method. Dataset names and files can remain in their existing local folders; this repository contains no dataset.

## Requirements and installation

Python 3.8 or later is supported. The method uses OpenCV, NumPy, and SciPy and runs on a CPU; TensorFlow, PyTorch, and GPU libraries are not required. The verified environment was Python 3.8.8, NumPy 1.24.4, SciPy 1.10.1, and OpenCV 4.11.0.86.

On Windows PowerShell:

    py -3 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

On macOS or Linux:

    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt

The commands below use the virtual environment's Python. In Windows PowerShell, activate it with .\.venv\Scripts\Activate.ps1; on macOS/Linux, use `source .venv/bin/activate`. If activation is unavailable, run .\.venv\Scripts\python.exe directly on Windows or `.venv/bin/python` on macOS/Linux.

## Input layout

Provide one normal template and one current inspection image for each matching file stem. For example, if your local data folders are named A4 and A5:

    local-data/
      A4/
        1.jpg
        2.jpg
      A5/
        1.jpg
        2.jpg
      A5_resized_xml/
        1.xml
        2.xml

The actual data are not distributed. Image IDs and Pascal VOC XML filenames must match. Ground-truth coordinates must use the same resized image coordinate system as the inference outputs. The evaluator reads XML boxes directly and does not modify coordinate origins.

## Run inference

From the repository root:

    python scripts/run_detection.py --template-dir "D:/path/to/local-data/A4" --inspection-dir "D:/path/to/local-data/A5" --output-json outputs/predictions.json --log-file outputs/run.log

Inference resizes each pair to 1080 × 720 by default. Prediction boxes use XYXY coordinates in that resized image space. The output JSON contains one selected box or a null prediction per image. Failed image pairs are recorded as errors and the command returns a non-zero exit code; failed rows must be resolved before evaluation.

To change parameters, edit configs/default.json or pass another JSON file with --config.

## Evaluate

Use the prediction JSON and the local XML annotation directory:

    python scripts/evaluate.py --predictions outputs/predictions.json --ground-truth-xml-dir "D:/path/to/local-data/A5_resized_xml" --iou-threshold 0.4 --missing-xml-is-negative --output-json outputs/metrics.json

A prediction is a true positive only when its IoU is strictly greater than 0.4 and it is matched one-to-one with a ground-truth box. Use --missing-xml-is-negative only when an absent XML file explicitly means that the image contains no ground-truth object. Otherwise, provide an XML file for every image, including empty files for normal images. Unmatched predictions count as false positives; unmatched ground-truth boxes count as false negatives. A normal image with no prediction is counted as a true negative. Precision and recall follow the formulas in the paper. The evaluator accepts any matching number of prediction and annotation IDs; it does not assume 100 samples.

## Run tests

From the repository root, run the focused unit tests with:

    python -m unittest discover -s tests -v

The tests cover grid construction, HaarPSI behavior, invalid warped regions, one-to-one IoU matching, missing-annotation handling, and a synthetic end-to-end image pair. They do not require the private inspection data.

## Method profiles and paper alignment

The default profile reproduces the independently audited source behavior: 1080 × 720 resizing; 8 × 8, 9 × 9, and 10 × 10 grids; AKAZE matching with OpenCV's default L2 distance; at least five ratio-test matches; ratio threshold 0.55; RANSAC reprojection threshold 4 pixels; and three-channel image arrays passed unchanged to HaarPSI. For repeatability, it uses the per-pair RANSAC seed introduced in the independent audit; the older source script did not set a seed. HaarPSI uses C = 30 and alpha = 4.2. A local cell is retained only when its HaarPSI score is below 0.9 times the full-image score and its black-pixel fraction is below 3%.

The manuscript instead specifies Hamming distance and grayscale HaarPSI. Run the literal-text profile explicitly with `--config configs/manuscript_literal.json`. On the checked local 100-pair set, that profile does not reproduce the manuscript's headline result. The manuscript does not specify the ratio threshold, RANSAC threshold, or minimum match count; these values come from the source/audit implementation. The per-image RANSAC seed was added during the independent audit and is not specified in the manuscript.

There is a real method-description discrepancy: the current manuscript says Hamming/grayscale, while the audited source execution uses OpenCV's default L2 matcher and passes three-channel OpenCV BGR arrays unchanged to the color HaarPSI path. The manuscript also says invalid black warp regions are excluded from similarity computation. This release excludes cells with at least 3% black pixels from candidate scoring, but the full-image baseline score still includes warp borders. Reconcile these method details with the implementation before claiming that every detail is identical. See PAPER_ALIGNMENT.md for the evidence.

The default profile was run on the 100 available image pairs and matched every saved independently audited prediction box. Under strict one-to-one IoU matching it measured TP/FP/FN/TN = 45/4/5/49, precision 91.84%, and recall 90.00%. The literal-text profile measured TP/FP/FN/TN = 35/6/15/47, precision 85.37%, and recall 70.00%. The second profile follows the current manuscript's stated matching distance and grayscale HaarPSI input; it is not the profile that reproduces the headline metrics. Neither profile includes the private image data.

## Scope limits

This is a still-image-pair detector, not a live webcam/video application. Neither this manuscript nor the `xlw` source tree contains point forecasting, prediction correction, prediction intervals, or an alert workflow. The source tree does contain YOLACT webcam/video demo code, but it is unrelated to this paper's proposed method and is excluded here. If those forecasting/alert features belong to another project, their actual source files are needed before they can be preserved.

The paper's supervised comparison models and their model-specific training/inference pipelines are not included, so this repository cannot reproduce all comparison tables. It includes a general-purpose evaluator and releases the proposed training-free method.

## Privacy and generated files

Do not commit operational images, XML annotations, prediction JSON files, model weights, logs, archives, or generated results. `.gitignore` excludes common data, output, log, archive, and model-file patterns. Check `git status` before committing, especially when saving files outside the ignored directories. Project-authored files contain no private contact details for the study authors or workstation-specific paths; required third-party attribution is retained in `THIRD_PARTY_NOTICES.md`.

## Citation and code availability

Please cite the HaarPSI paper described in the manuscript when using the metric. This repository includes the NumPy HaarPSI implementation and preserves its upstream attribution and MIT license in THIRD_PARTY_NOTICES.md.

The root `LICENSE` currently contains a draft MIT grant. MIT permits modification, commercial reuse, redistribution, and sublicensing. Before publishing, the corresponding author and other rights holders must confirm that they authorize this license for all included original code and finalize the copyright notice. Do not treat the presence of the file as evidence that this approval has been obtained.

After publishing this repository, replace the manuscript's current “code available from the corresponding author upon reasonable request” wording with the public repository URL, while keeping the data unavailability and access restrictions accurate.
