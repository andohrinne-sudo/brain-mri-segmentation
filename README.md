# Brain MRI Segmentation: A 3D U-Net Approach

This project presents a robust solution for automated brain tumor segmentation using a 3D U-Net architecture. Designed for clinical research and application, it accurately delineates multiple tumor sub-regions from volumetric MRI data, integrating advanced deep learning techniques with MLOps best practices for efficient and reliable deployment.

## Model Architecture: 3D U-Net for Volumetric Data

The core of this project is a sophisticated 3D U-Net model, specifically engineered to process the inherent three-dimensional nature of MRI scans. This architecture enables the extraction of rich spatial features crucial for precise tumor localization.

*   **Input Handling**: The model is configured to accept input in `channels_first` format, with an `input_shape` of `(4, 160, 160, 16)`. This design efficiently handles multi-sequence MRI data (e.g., T1, T1ce, T2, FLAIR) and processes 3D volumetric patches.

*   **Encoder Path (Feature Extraction)**: The encoder path consists of successive blocks of:
    *   `Conv3D` layers: These layers apply 3D convolutions to extract features across all three spatial dimensions (width, height, depth) of the MRI volume. Each block typically contains two `Conv3D` layers followed by a ReLU activation.
    *   `MaxPooling3D` layers: These downsample the feature maps, effectively increasing the receptive field and reducing the spatial resolution while preserving essential high-level features. This step helps in learning more abstract representations of the tumor.

*   **Bottleneck**: A central block of `Conv3D` layers connects the encoder and decoder, capturing the most abstract and contextual features from the downsampled data.

*   **Decoder Path (Spatial Localization)**: The decoder path symmetrically reconstructs the segmentation map, integrating fine-grained details:
    *   `UpSampling3D` layers: These layers increase the spatial dimensions, gradually restoring the original resolution.
    *   **Skip Connections**: A critical component of the U-Net, skip connections concatenate (along the channel axis) the high-resolution feature maps from the corresponding encoder levels with the upsampled features in the decoder. This mechanism ensures that fine-grained spatial information, lost during downsampling, is re-integrated, enabling highly precise localization of tumor boundaries.

*   **Output Layer**: The final `Conv3D` layer outputs `n_labels` (typically 3 for 'Edema', 'Non-Enhancing Tumor Core', 'Enhancing Tumor') feature maps, each followed by a `sigmoid` activation function. This produces independent probability maps for each tumor sub-region, allowing for multi-class, multi-label segmentation.

## Multi-Class Sub-Region Segmentation

This model is specifically designed for multi-class sub-region segmentation, providing granular detail beyond simple tumor detection. It distinguishes between the following clinically significant tumor components:

*   **Edema**
*   **Non-Enhancing Tumor Core**
*   **Enhancing Tumor**

The `sigmoid` activation in the final layer facilitates the prediction of independent probabilities for each class at every voxel, making the model adept at handling complex, potentially overlapping tumor morphologies.

## Addressing Class Imbalance: Soft Dice Loss

Medical image segmentation, especially for brain tumors, frequently encounters severe class imbalance, where the tumor regions occupy a significantly smaller volume than healthy tissue. To effectively address this, a **Soft Dice Loss** function is utilized during model training.

*   **Purpose**: Unlike traditional loss functions (e.g., binary cross-entropy) that can be overwhelmed by the abundance of negative (healthy tissue) samples, Soft Dice Loss directly optimizes for the Dice coefficient, a measure of spatial overlap between the predicted and true segmentation masks.
*   **Mechanism**: It specifically promotes better spatial overlap for smaller, harder-to-segment tumor regions, assigning them greater importance during training. This ensures that the model learns to accurately delineate even sparse tumor sub-regions, which is critical for clinical utility and accurate diagnosis.

## MLOps Considerations: 3D Patch Extraction for GPU Memory and Local Structure Preservation

The implementation incorporates key MLOps considerations to ensure the model's practicality and performance in real-world clinical environments:

*   **Efficient Data Loading**: The `MRIDataLoader` class (`dataloader.py`) supports diverse input formats (NifTI and HDF5) and extracts 3D patches.

*   **3D Patch Extraction Strategy**: The use of `160x160x16` 3D patches for both training and inference (via a **sliding window approach** in `predict_full_volume` from `mrisegmentor.py`) is a deliberate MLOps decision. This strategy addresses two critical challenges:
    1.  **GPU Memory Management**: Full 3D MRI volumes are often too large to fit into GPU memory. By processing data in smaller, fixed-size patches, the model can be trained and deployed efficiently on standard GPU hardware without prohibitive memory requirements.
    2.  **Local Structure Preservation**: Processing data in patches allows the model to focus on intricate local anatomical details and tumor structures within each sub-volume. This approach helps preserve the fine-grained contextual information crucial for accurate segmentation, as the model's convolutional kernels operate within a relevant local neighborhood.

*   **Standardized Preprocessing**: Consistent **zero-mean unit-variance (Z-score) normalization** is applied to all patches across both `dataloader.py` and `mrisegmentor.py`. This standardization is vital for model stability, faster convergence, and robust performance, especially when using pre-trained weights or transfer learning concepts.

## Clinical Evaluation Metrics

To rigorously assess the model's diagnostic accuracy and ensure its reliability in a clinical context, the `compute_metrics` method (`mrisegmentor.py`) calculates essential metrics for each tumor sub-region:

*   **Sensitivity (Recall)**: Measures the proportion of actual positive cases (tumor presence) that are correctly identified. High sensitivity is paramount in medical diagnosis to minimize false negatives and avoid missing critical tumor findings.
*   **Specificity**: Measures the proportion of actual negative cases (healthy tissue) that are correctly identified. High specificity is crucial for minimizing false positives, thereby reducing unnecessary follow-up examinations and patient anxiety.

These metrics provide a balanced and clinically relevant view of the model's performance, addressing both the detection of disease and the avoidance of misdiagnosis.

## Setup & Run

### Prerequisites
*   Python 3.8+
*   `pip` package manager

### Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/andohrinne-sudo/brain-mri-segmentation.git
    cd brain-mri-segmentation
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Model Weights**: Ensure that the pre-trained model weights (`model_pretrained.hdf5`) are present in the root directory of the cloned repository. These weights are essential for the `MRISegmentor` to load a functional model.

### Running the Application

To execute the main application logic (e.g., for inference or further development):

```bash
python main.py
```

*(Note: This project focuses on the core segmentation logic. A Streamlit or similar interactive application would typically wrap `main.py` for a full MLOps deployment scenario.)*
