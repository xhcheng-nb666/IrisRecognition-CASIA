# Iris Recognition Using CASIA-IrisV1

## 1. Project Overview

This project implements an iris recognition system based on the classical image-processing and machine-learning methods described in Ma et al. (2003), *Personal Identification Based on Iris Texture Analysis*.

The system processes iris images, extracts discriminative texture features, and performs iris identification and verification.

The complete recognition pipeline consists of six stages:

1. Iris Localization
2. Iris Normalization
3. Image Enhancement
4. Feature Extraction
5. Iris Matching
6. Performance Evaluation

The implementation uses Python, OpenCV, NumPy, SciPy, and scikit-learn.

The overall design follows Ma et al. (2003), with some practical adaptations described in Section 5.

## 2. Dataset and Experimental Design

The CASIA Iris Image Database Version 1.0 (CASIA-IrisV1) is used for this project.

The dataset contains:

- 108 iris classes (eyes)
- 7 images per class
- 756 grayscale BMP images
- Image resolution: 320 × 280 pixels

Each iris class contains images captured during two sessions.

Following the assignment requirements, the dataset is divided into:

| Dataset | Session | Images per Class | Total Images |
|---|---|---|---|
| Training | Session 1 | 3 | 324 |
| Testing | Session 2 | 4 | 432 |

Only session 1 images are used for model training and enrollment. Session 2 images are used exclusively for testing.

This separation ensures that the recognition system is evaluated using images captured in a different session from the training images.

## 3. System Design and Implementation

### 3.1 Iris Localization (`IrisLocalization.py`)

The purpose of iris localization is to detect the pupil boundary and the outer iris boundary.

The implementation first applies Gaussian smoothing to reduce image noise.

Because CASIA-IrisV1 contains artificially darkened pupil regions, intensity thresholding is used to detect dark regions. Connected-component analysis identifies potential pupil candidates based on their area, shape, and position.

If this method fails, the Hough Circle Transform is used as a fallback.

After detecting the pupil, the outer iris boundary is estimated using Sobel gradient magnitudes and a circular boundary search.

The search focuses on the left and right portions of the iris to reduce interference from eyelids.

The output consists of two circles representing the pupil and outer iris boundaries.

This localization method is an alternative to the original paper's procedure, as permitted by the assignment.

### 3.2 Iris Normalization (`IrisNormalization.py`)

Iris normalization converts the circular iris region into a rectangular representation.

The system uses a rubber-sheet transformation to map pixels between the pupil boundary and outer iris boundary into polar coordinates.

Each normalized iris image has dimensions:

**64 × 512 pixels**

The radial direction contains 64 samples, while the angular direction contains 512 samples.

This transformation reduces geometric differences caused by variations in pupil size and iris position.

The implementation also supports angular offsets, which are used for rotation compensation during matching.

### 3.3 Image Enhancement (`ImageEnhancement.py`)

Image enhancement improves iris texture visibility by reducing uneven illumination and increasing local contrast.

The implementation consists of three steps.

**Step 1: Background Illumination Estimation**

The normalized image is divided into 16 × 16 regions.

The average intensity of each region is calculated to estimate background illumination.

Bicubic interpolation is then used to construct a full-resolution illumination estimate.

**Step 2: Illumination Correction**

The estimated background is subtracted from the normalized iris image.

A constant offset is added, and the pixel values are clipped to the valid grayscale range.

**Step 3: Local Histogram Equalization**

Histogram equalization is applied independently to 32 × 32 regions.

This enhances local texture patterns and prepares the image for feature extraction.

### 3.4 Feature Extraction (`FeatureExtraction.py`)

Feature extraction converts the enhanced iris image into a numerical feature vector.

Following Ma et al. (2003), the implementation uses the first 48 rows of the 64 × 512 enhanced image.

Two Gaussian-modulated circular spatial filters are applied to capture iris texture information at different spatial scales.

The filter parameters are:

- Channel 1: σx = 3.0, σy = 1.5
- Channel 2: σx = 4.5, σy = 1.5

The implementation uses a configurable modulation frequency, with a default value of 0.15.

After filtering, the absolute filter responses are divided into nonoverlapping 8 × 8 blocks.

For each block, two statistical features are calculated:

1. Mean intensity
2. Mean absolute deviation

Each filter channel produces:

48 / 8 × 512 / 8 = 384 blocks

With two statistical features per block and two filter channels, the final feature-vector dimension is:

384 × 2 × 2 = **1,536 features**

These feature vectors represent the local texture characteristics of each iris.

### 3.5 Iris Matching (`IrisMatching.py`)

The iris matching module performs dimensionality reduction and identity classification.

**Step 1: Feature Standardization**

The extracted features are standardized using `StandardScaler`.

This reduces differences in feature scales before dimensionality reduction.

**Step 2: PCA**

Principal Component Analysis is applied to reduce the dimensionality of the original feature vectors.

The implementation retains up to 100 principal components.

This preliminary reduction helps address numerical difficulties caused by the high-dimensional feature space and limited number of training samples.

**Step 3: Fisher Linear Discriminant Analysis**

Fisher Linear Discriminant Analysis (LDA) is applied to obtain discriminative features.

LDA aims to maximize differences between iris classes while minimizing variation within each class.

The implementation uses regularized Fisher LDA with a shrinkage parameter of 0.1.

**Step 4: Nearest-Center Classification**

For each iris class, a feature center is calculated by averaging the projected training features belonging to that class.

A testing image is classified by finding the nearest class center.

Three distance measures are evaluated:

- L1 (Manhattan distance)
- L2 (Euclidean distance)
- Cosine distance

The predicted identity is the class associated with the minimum distance.

**Step 5: Rotation Compensation**

To compensate for possible angular differences between iris images, seven rotation angles are considered:

-9°, -6°, -3°, 0°, 3°, 6°, 9°

Each training image is normalized using these seven angular offsets.

The resulting feature vectors are projected into the trained PCA–LDA space.

For each iris class, seven rotation-specific class centers are calculated.

During matching, the testing image is compared with all seven centers for each class.

The minimum distance across the seven rotations is selected as the final distance to that class.

This improves robustness to rotational differences between iris images.

The dimensionality-reduction models are trained only on the original session 1 training samples. Rotated training features are subsequently projected using those fitted models.

### 3.6 Performance Evaluation (`PerformanceEvaluation.py`)

The system evaluates two recognition modes.

**Identification Mode**

In identification mode, the system predicts which of the 108 enrolled iris classes corresponds to each testing image.

Performance is measured using the Correct Recognition Rate (CRR):

CRR = (Number of Correct Predictions / Total Testing Images) × 100%

CRR is calculated separately for L1, L2, and Cosine distance.

The assignment requires all three CRR values to be at least 80%.

**Verification Mode**

In verification mode, the system evaluates whether a testing iris matches a claimed identity.

Each testing feature vector is compared against the 108 enrolled class centers, using rotation-compensated distances.

Genuine comparisons correspond to the correct iris identity, while impostor comparisons correspond to different identities.

The following metrics are calculated:

- False Match Rate (FMR)
- False Nonmatch Rate (FNMR)
- Receiver Operating Characteristic (ROC)
- Area Under the ROC Curve (AUC)
- Approximate Equal Error Rate (EER)

The implementation generates CSV tables and figures summarizing identification and verification performance.

## 4. Experimental Results

The system was evaluated using all 432 testing images from session 2.

### 4.1 Identification Results

| Distance Measure | Correct Predictions | Total | CRR |
|---|---|---|---|
| L1 | 405 | 432 | 93.75% |
| L2 | 402 | 432 | 93.06% |
| Cosine | 412 | 432 | 95.37% |

All three measures exceed the required 80% CRR.

Cosine distance achieves the highest identification accuracy of 95.37%.

### 4.2 Verification Results

| Distance Measure | AUC | Approximate EER |
|---|---|---|
| L1 | 0.9766 | 6.96% |
| L2 | 0.9831 | 5.36% |
| Cosine | 0.9942 | 2.60% |

Cosine distance provides the best verification performance, achieving the highest AUC and lowest approximate EER.

The results suggest that Cosine distance is more effective than L1 and L2 for this implementation.

### 4.3 Rotation Compensation

Compared with the original implementation without rotation compensation, the updated system improves identification accuracy.

| Distance Measure | Without Rotation | With Rotation |
|---|---|---|
| L1 | 89.81% | 93.75% |
| L2 | 90.97% | 93.06% |
| Cosine | 94.91% | 95.37% |

Rotation compensation improves CRR across all three distance measures.

This indicates that accounting for angular differences can improve recognition accuracy on CASIA-IrisV1.

## 5. Limitations

Although the system achieves recognition accuracy above the assignment's minimum requirement, several limitations remain.

**1. Iris Localization Accuracy**

The localization algorithm assumes approximately circular pupil and iris boundaries.

It may perform less reliably when iris boundaries are partially obscured by eyelashes, eyelids, or strong reflections.

The fixed radius search ranges may also fail for images with unusual iris sizes or positions.

**2. Simplified Feature Extraction Parameters**

Although the implementation follows the two-channel spatial-filtering approach of Ma et al. (2003), some parameters, including the finite kernel dimensions and modulation frequency, are implementation choices.

These settings may affect the extracted features and recognition accuracy.

**3. Modified Fisher LDA Procedure**

The original paper emphasizes Fisher Linear Discriminant Analysis for dimensionality reduction.

Our implementation additionally uses feature standardization, PCA, and regularization before and during LDA.

These modifications improve numerical stability but mean that the implementation is not an exact reproduction of the original dimensionality-reduction procedure.

**4. Rotation Matching Approximation**

The implementation generates seven rotation-specific class centers and selects the minimum distance during matching.

Although this follows the general concept of rotation compensation, it is an adaptation of the paper's rotational template-matching procedure.

**5. Limited Dataset Diversity**

CASIA-IrisV1 contains only 108 iris classes captured under controlled conditions.

Therefore, the reported performance may not generalize to larger datasets or images captured under more challenging real-world conditions.

**6. Verification Protocol**

Verification is evaluated using testing samples compared against enrolled class centers.

This differs from a protocol based on all possible individual image pairs and may produce different ROC results.

Therefore, the reported results should not be treated as directly equivalent to all experimental results in the original paper.

## 6. Possible Improvements

Several improvements could be considered in future work.

**1. More Robust Iris Segmentation**

More advanced boundary-detection methods could improve localization accuracy, especially when eyelids, eyelashes, or reflections partially obscure the iris.

Noise masking could also be introduced to exclude unreliable iris regions from feature extraction.

**2. Improved Feature Extraction**

The spatial-filter parameters could be further verified against the original paper.

Additional experiments could evaluate how different modulation frequencies and kernel sizes affect recognition accuracy.

**3. More Faithful Fisher LDA Implementation**

The dimensionality-reduction stage could be revised to reproduce the paper's Fisher LDA procedure more directly.

This would reduce methodological differences between the implementation and the original research.

**4. Improved Rotation Compensation**

Instead of using seven rotation-specific class centers, individual rotated templates could be retained and matched using a procedure more closely aligned with the original paper.

**5. Evaluation on Additional Datasets**

The recognition system could be evaluated on newer iris datasets containing more identities and greater variation in image quality.

This would provide stronger evidence of generalization.

**6. Computational Efficiency**

Although extracted features are cached to avoid redundant calculations, rotation compensation increases preprocessing time and storage requirements.

More efficient feature extraction and template processing could reduce computational cost.

## 7. Project Files and Execution

The main project files are:

```text
IrisRecognition.py
IrisLocalization.py
IrisNormalization.py
ImageEnhancement.py
FeatureExtraction.py
IrisMatching.py
PerformanceEvaluation.py
requirements.txt
```

The dataset should be extracted into the `data` directory.

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the Recognition System

```bash
python IrisRecognition.py --data data --out results_rotation
```

### Recompute All Features

```bash
python IrisRecognition.py --data data --out results_rotation --recompute
```

The `--recompute` option forces the system to regenerate features instead of loading previously cached values.

### Disable Rotation Compensation

```bash
python IrisRecognition.py --data data --out results_baseline --no-rotation
```

### Output Files

The program generates the following experimental results:

```text
results_rotation/
    Table3_CRR.csv
    Table4_Verification.csv
    Figure10_CRR.png
    Figure11_ROC.png
    Figure11_FMR_FNMR.png
    ROC_L1.csv
    ROC_L2.csv
    ROC_cosine.csv
```

The program also stores intermediate feature vectors in `.npz` files to reduce repeated preprocessing.

## 8. References

[1] L. Ma, T. Tan, Y. Wang, and D. Zhang, "Personal Identification Based on Iris Texture Analysis," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 25, no. 12, pp. 1519–1533, 2003.

[2] Chinese Academy of Sciences Institute of Automation (CASIA), "CASIA Iris Image Database Version 1.0 (CASIA-IrisV1)."

Portions of this project use CASIA-IrisV1 collected by the Chinese Academy of Sciences' Institute of Automation (CASIA).