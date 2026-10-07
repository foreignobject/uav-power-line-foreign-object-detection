# Manuscript and Source-Code Audit

The high-level IFM-MGS-SAM method is consistent between the manuscript and this release. This audit focuses on two implementation settings in the submitted draft (matching distance and HaarPSI input channels) and on the exact scope of the invalid-pixel rule. The release preserves the settings that reproduce the saved, independently audited predictions and reported headline metrics.

## What the manuscript describes

The proposed method is a training-free detector for paired still images. It aligns a historical normal template and a current inspection image with AKAZE/MLDB feature matching and a RANSAC homography (IFM), compares corresponding 8 x 8, 9 x 9, and 10 x 10 image grids (MGS), and selects the lowest-scoring eligible region using HaarPSI (SAM). The manuscript specifies Hamming distance for binary descriptors, grayscale HaarPSI equations, a local-score condition below 0.9 times the full-image score, a 3% invalid-black-pixel rule, and IoU > 0.4 for evaluation.

The proposed IFM-MGS-SAM method has no model-training stage. The manuscript describes training/validation of separate supervised comparison detectors using labeled images; those baseline training pipelines are not part of this release.

## Two executable profiles

| Profile | Configuration | IFM matching | HaarPSI input | TP / FP / FN / TN | Precision / recall |
| --- | --- | --- | --- | --- | --- |
| Audited source behavior | `configs/default.json` | AKAZE descriptors with OpenCV L2; ratio < 0.55; at least 5 matches; RANSAC threshold 4 px | Three-channel OpenCV BGR array passed unchanged to HaarPSI | 45 / 4 / 5 / 49 | 91.84% / 90.00% |
| Manuscript-described Hamming/grayscale behavior; unspecified thresholds inherited from source | `configs/manuscript_literal.json` | AKAZE descriptors with Hamming; source-inherited ratio < 0.55, at least 5 matches, and RANSAC threshold 4 px | Grayscale | 35 / 6 / 15 / 47 | 85.37% / 70.00% |

The manuscript specifies Hamming distance and grayscale HaarPSI, but it does not specify the ratio threshold, minimum match count, or RANSAC reprojection threshold. Those three settings in the literal-text profile are inherited from the audited source implementation; they are not stated manuscript parameters.

Both profiles were run on the same 100 local image pairs, with no failed inference rows. For this dataset, 50 positive images have XML files containing 50 ground-truth boxes, while the 50 normal images have no XML file. Evaluation therefore used the explicit `--missing-xml-is-negative` option. Both profiles use strict IoU > 0.4 and one-to-one matching; every unmatched prediction is an FP and every unmatched ground-truth box is an FN.

The audited source-behavior profile matched all 100 saved per-image boxes in `xlw/scripts/final_independent_audit.py`'s existing audit output. Maximum absolute score differences were below 5e-13. This independently reproduces the saved audit result, but does not prove which exact source revision produced the manuscript's original experiment. The RANSAC seed by image ID was added during that independent audit; the older source script did not set a seed.

## Implementation details to correct in the manuscript

The manuscript says Hamming matching, but the original `similarity_hmy.py` calls `cv2.BFMatcher()` without a norm argument. OpenCV's default is L2. It also passes three-channel arrays to HaarPSI after calling `cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)` on three-channel inputs. In the installed OpenCV 4.11 environment, this conversion leaves those three-channel arrays unchanged; they therefore reach HaarPSI in the original BGR channel order and use its color path. The saved independent audit uses the same L2 and three-channel behavior, and this release reproduces its boxes with that profile.

The manuscript's grayscale/Hamming profile instead produces 85.37% precision and 70.00% recall on the same private test set. The release therefore includes both behaviors explicitly rather than claiming they are equivalent. Before public submission, the authors should decide whether the method text should be corrected to describe the implementation behind 91.84%/90.00%, or whether the method should follow the written Hamming/grayscale specification and the reported performance should be recomputed.

The manuscript describes regions with at least 3% black pixels as excluded from similarity assessment. This release checks the black-pixel condition before computing local HaarPSI for such cells; this does not change which cells can be retained. The full-image HaarPSI still includes black warp borders. If the manuscript intends those pixels to be masked out of the global score too, that requires a different method definition and another evaluation.

The original evaluator in `similarity_hmy.py` does not count a missed ground-truth box as an FN when a prediction exists but fails the IoU threshold. This can overstate recall. The release evaluator uses the one-to-one rule described above. The published 91.84%/90.00% is consistent with the independently audited TP=45, FP=4, FN=5 counts under the corrected rule.

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
- The audited-source profile processed 100/100 pairs and matched all 100 saved audit prediction boxes exactly.
- The manuscript-literal profile processed 100/100 pairs and produced TP/FP/FN/TN = 35/6/15/47.
- No images, XML labels, model weights, author identity, or workstation paths are included in the release source files. The dataset remains private.
- HaarPSI's attribution and third-party MIT terms are retained in `THIRD_PARTY_NOTICES.md`. The root MIT license is for the release authors to approve collectively before publication.
