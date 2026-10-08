# Manuscript and Source-Code Alignment

The revised manuscript describes the released IFM-MGS-SAM reference profile. This note records the implementation details that were corrected in the method text and the remaining limit on historical provenance.

## Method described in the revised manuscript

The proposed method is a training-free detector for paired still images. It resizes each pair to 1080 × 720, aligns a historical normal template and a current inspection image with AKAZE/MLDB feature matching and a RANSAC homography (IFM), compares corresponding 8 × 8, 9 × 9, and 10 × 10 image grids (MGS), and selects the lowest-scoring eligible region using HaarPSI (SAM). The revised method text specifies OpenCV L2 matching, a ratio threshold of 0.55, at least five retained matches, a 4-pixel RANSAC reprojection threshold, three-channel BGR input passed unchanged to HaarPSI, and the color component of the HaarPSI score. The local-score condition is below 0.9 times the full-image score. A local inspection crop is skipped before local HaarPSI if at least 3% of its pixels are exactly zero in all BGR channels; the full-image score includes warp borders, and no pixel mask is applied to eligible crops.

The proposed IFM-MGS-SAM method has no model-training stage. The manuscript describes training and validation of separate supervised comparison detectors; those baseline training pipelines are not part of this release.

## Audited profiles and reported results

| Profile | Configuration | IFM matching | HaarPSI input | TP / FP / FN / TN | Precision / recall |
| --- | --- | --- | --- | --- | --- |
| Revised-manuscript reference profile | `configs/default.json` | AKAZE descriptors with OpenCV L2; ratio < 0.55; at least 5 matches; RANSAC threshold 4 px | Three-channel OpenCV BGR array passed unchanged to HaarPSI | 45 / 4 / 5 / 49 | 91.84% / 90.00% |
| Historical draft-text comparison | `configs/manuscript_literal.json` | AKAZE descriptors with Hamming; ratio < 0.55; at least 5 matches; RANSAC threshold 4 px | Grayscale | 35 / 6 / 15 / 47 | 85.37% / 70.00% |

The Hamming/grayscale profile preserves the earlier draft's wording for comparison; it is not described as equivalent to the revised method. Both saved profiles were evaluated on the same 100 local image pairs. The dataset has 50 positive images with 50 ground-truth boxes and 50 normal images without XML files, so evaluation used the explicit `--missing-xml-is-negative` option. Both use strict IoU > 0.4 and one-to-one matching; unmatched predictions are false positives and unmatched ground-truth boxes are false negatives.

The default release profile matched all 100 saved per-image boxes in `xlw/scripts/final_independent_audit.py`'s existing audit output, with maximum absolute score differences below 5e-13. The 91.84% precision and 90.00% recall correspond to TP/FP/FN = 45/4/5 under the corrected evaluator. This independently reproduces the saved audit outputs; it does not prove which exact source revision or RANSAC seed produced the original manuscript experiment. The pair-ID RANSAC seed was added during the independent audit; the older source script did not set a seed. The revised manuscript states this distinction explicitly.

## Key implementation facts now stated in the manuscript

- `cv2.BFMatcher()` in the older source used OpenCV's default L2 norm; the release names that norm explicitly.
- OpenCV image arrays enter the released HaarPSI path in BGR order. The upstream routine applies its YIQ coefficients to the three channels as received, so the revised text documents that exact channel behavior rather than calling it grayscale or standard RGB.
- The HaarPSI color score includes two Y-component orientation maps plus a combined I/Q map. The revised equations define the additional similarity and its weight.
- The 3% condition checks exactly black pixels in the inspection crop and skips the whole crop before local HaarPSI. It neither masks black pixels from eligible crops nor removes them from the full-image score.
- The release uses a stable pair-ID RANSAC seed for reproducible reruns. The available historical source cannot establish whether that seed was used for the original published metrics.
- Evaluation uses strict one-to-one IoU matching at IoU > 0.4; unmatched predictions count as false positives and unmatched ground-truth boxes count as false negatives. The reported 91.84%/90.00% was recomputed from the saved audited predictions with this rule. The older source evaluator did not consistently count every unmatched ground-truth box as a false negative.

## Scope review against the original `xlw` tree

- `xlw` contains no point forecasting, prediction correction, prediction intervals, or alert workflow, and the manuscript does not describe such functions. They are not implemented in this release; they cannot be safely reconstructed by renaming unrelated functions.
- `xlw/segmentation/train.py` trains a YOLACT instance-segmentation branch for cable/tower classes. `eval.py` and `eval1.py` include image/video/webcam demos. This is a separate project branch; YOLACT is not one of this manuscript's listed comparison methods, so it is excluded from this paper's release.
- The `Interval3` token found in a YOLACT backbone/configuration name is not an interval-prediction implementation. Detection confidence thresholds and NMS are object-detection post-processing, not prediction correction or an alert lifecycle.
- The manuscript's supervised baselines are separate from the proposed training-free method. Their full training/inference configurations and reproducible checkpoints are not present as a coherent package in the selected release scope. This repository does not claim to recreate the manuscript's full comparison tables; it provides the proposed method and its generic evaluator.
- The release is offline paired-image inference, not a live-stream application. It outputs at most one candidate box for each template/inspection pair.

If point prediction, interval prediction, corrections, and alarms are required for a different application, the source directory and corresponding paper for that application are needed. The current paper and `xlw` contents do not support that scope.

## Verification performed

- Six focused unit tests pass, including grid construction, HaarPSI, invalid-black-cell short-circuiting, unmatched-box accounting, explicit missing-XML behavior, and a synthetic end-to-end image pair. The verified environment was Python 3.8.8, NumPy 1.24.4, SciPy 1.10.1, and OpenCV 4.11.0.86.
- `run_detection.py --help` and `evaluate.py --help` run successfully.
- The default release profile matched all 100 saved audit prediction boxes exactly; the Hamming/grayscale comparison profile produced TP/FP/FN/TN = 35/6/15/47 on the same local pairs.
- No images, XML labels, model weights, author identity, or workstation paths are included in the release source files. The dataset remains private.
- All relevant rights holders approved the root MIT license for the included original code. HaarPSI's attribution and separate upstream MIT terms are retained in `THIRD_PARTY_NOTICES.md`.
