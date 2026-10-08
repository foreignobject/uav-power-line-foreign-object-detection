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

The revised manuscript describes the released IFM-MGS-SAM reference profile: AKAZE feature matching with OpenCV L2 distance, a 0.55 ratio threshold, at least five retained matches, a 4-pixel RANSAC threshold, three complementary grid scales, and HaarPSI on three-channel OpenCV BGR arrays passed without channel reordering. The full-image HaarPSI score includes black borders introduced by warping. Before local HaarPSI is computed, a grid cell is skipped when at least 3% of pixels in its inspection crop have all three channels exactly zero; retained cells are not pixel-masked.

The release defaults to 1080 × 720 images and 8 × 8, 9 × 9, and 10 × 10 grids. HaarPSI uses C = 30 and alpha = 4.2 with its default factor-two subsampling. The public implementation also seeds RANSAC from the stable image-pair identifier. That seed policy was added for repeatability during the independent audit; the archived original script does not set an explicit seed, so the release setting is not claimed as a verified historical run parameter.

The revised method description aligns with this public reference profile. `configs/manuscript_literal.json` preserves the earlier Hamming/grayscale interpretation as a historical comparison profile; it is not the method associated with the reported headline result. The manuscript does not specify the ratio threshold, minimum match count, or RANSAC threshold, so those implementation values are stated explicitly in the revised method text.

The default release profile matched all 100 saved independently audited prediction boxes. Re-evaluation under strict one-to-one IoU matching gave TP/FP/FN/TN = 45/4/5/49, precision 91.84%, and recall 90.00%. The Hamming/grayscale comparison profile gave TP/FP/FN/TN = 35/6/15/47, precision 85.37%, and recall 70.00%. These saved predictions and image data are private and are not included in this repository.

## Scope limits

This is a still-image-pair detector, not a live webcam/video application. Neither this manuscript nor the `xlw` source tree contains point forecasting, prediction correction, prediction intervals, or an alert workflow. The source tree does contain YOLACT webcam/video demo code, but it is unrelated to this paper's proposed method and is excluded here. If those forecasting/alert features belong to another project, their actual source files are needed before they can be preserved.

The paper's supervised comparison models and their model-specific training/inference pipelines are not included, so this repository cannot reproduce all comparison tables. It includes a general-purpose evaluator and releases the proposed training-free method.

## Privacy and generated files

Do not commit operational images, XML annotations, prediction JSON files, model weights, logs, archives, or generated results. `.gitignore` excludes common data, output, log, archive, and model-file patterns. Check `git status` before committing, especially when saving files outside the ignored directories. Project-authored files contain no private contact details for the study authors or workstation-specific paths; required third-party attribution is retained in `THIRD_PARTY_NOTICES.md`.

## Citation and code availability

Please cite the HaarPSI paper described in the manuscript when using the metric. This repository includes the NumPy HaarPSI implementation and preserves its upstream attribution and MIT license in THIRD_PARTY_NOTICES.md.

The project is distributed under the MIT License. All relevant rights holders have approved this license for the included original code. The collective copyright notice is `The Authors` to avoid exposing individual author identities. HaarPSI retains its upstream copyright and license in `THIRD_PARTY_NOTICES.md`.

The revised manuscript provides the public source-code URL: https://github.com/foreignobject/uav-power-line-foreign-object-detection. The UAV inspection dataset remains private and is not distributed in this repository.
