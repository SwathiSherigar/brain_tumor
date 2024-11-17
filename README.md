

# Brain Tumor Segmentation and Visualization

This project provides a web application built with **Streamlit** to perform **brain tumor segmentation** from MRI images. The app leverages a deep learning model to classify and segment different tumor regions in **FLAIR**, **T1CE**, and **segmentation masks** from MRI data. It visualizes the results and calculates tumor biomarkers such as tumor volume and tumor count.

## Features

- **Upload MRI Images**: Users can upload **FLAIR**, **T1CE**, and **segmentation mask** images in `.nii` (NIfTI) format.
- **Model Prediction**: The model segments the uploaded MRI images into four regions: **Necrotic/Core**, **Edema**, **Enhancing**, and **Non-Tumor**.
- **Tumor Biomarker Calculation**: The app calculates the voxel volume and tumor count for each class of the tumor (necrotic/core, edema, enhancing).
- **Visualization**: 
  - 2D slice-based display of MRI images and tumor segmentation.
  - Generation of a GIF from the 3D volume of the tumor.
  - Plotting of **FLAIR** images and segmentation masks using **Nilearn**.
  - Interactive plots with **Plotly** and **Mayavi** for 3D volume visualization.

## Requirements

This project requires Python and several Python libraries. You can set up the environment using the following `requirements.txt` file.

### `requirements.txt`

```txt
streamlit==1.21.0
keras==2.14.0
tensorflow==2.14.0
matplotlib==3.8.0
numpy==1.24.3
nibabel==5.2.0
opencv-python==4.8.0.74
plotly==5.11.0
mayavi==4.7.4
configobj==5.0.8
imageio==2.31.1
nilearn==0.9.0
gif-your-nifti==0.1.3
scipy==1.11.1
```

### To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## How to Use

### 1. Launch the Streamlit Application
To run the app locally, use the following command in your terminal:

```bash
streamlit run braintumor.py
```

This will start a local development server. Open the provided link (usually `http://localhost:8501`) in your browser to access the web interface.

### 2. Upload MRI Images
Once the app is running:
- Upload the **FLAIR Image (.nii)**, **T1CE Image (.nii)**, and **Segmentation Image (.nii)** using the provided file upload buttons.
  
### 3. View Predictions and Biomarkers
- The app will automatically perform the segmentation, show the predicted regions, and provide a breakdown of tumor biomarkers such as:
  - **Voxel count** (number of voxels in each tumor class).
  - **Tumor volume** in **mm³**.
  - **Tumor count** (number of connected components identified within each tumor class).

### 4. Visualize Tumor Regions
- The app displays 2D slices of the **FLAIR** image and segmented tumor regions.
- It also generates a **GIF** representation of the 3D volume of the tumor, which you can view directly in the app.

## Functions

### `predict_images(flair_file, t1ce_file, seg_file)`
- **Input**: Three NIfTI files (FLAIR, T1CE, and Segmentation).
- **Output**: Predictions of tumor segmentation and paths to the temporary image files.

### `display_predictions(predictions, flair_path, seg_path, start_slice=70, gt=None)`
- Displays the predicted tumor regions (necrotic/core, edema, enhancing) along with the **FLAIR** image and **segmentation mask**.

### `calculate_biomarkers(predictions, flair_path)`
- Calculates biomarkers such as voxel count, tumor volume, and tumor count for each tumor class based on model predictions.

### `visualize_t1_t2(t1_data, t2_data)`
- Displays the middle slice of the **T1** and **T2** weighted images for comparison.

### `create_gif_from_nifti(nifti_file, fps=5)`
- Converts a 3D **NIfTI** volume into a **GIF** animation of its slices.

### `plot_nifti_images(flair_file, seg_file)`
- Plots various representations of **FLAIR** images using **Nilearn**.

## File Formats Supported
- The app expects **NIfTI files (.nii)** as input for the MRI images. NIfTI is a common format for storing volumetric data such as MRI images.

## Model

The segmentation model is a pre-trained Keras model (`model_per_class.h5`). It performs multi-class segmentation for brain tumors, with the following classes:

1. **Necrotic/Core**
2. **Edema**
3. **Enhancing**
4. **Non-Tumor (Background)**

## Example Output

After uploading the MRI images and waiting for the model to process them, the app will display the following:
- Predicted tumor regions (necrotic/core, edema, enhancing) overlaid on the **FLAIR** image.
- Tumor biomarker information, including **volume** and **count** for each tumor region.
- A **GIF** of the tumor region’s 3D volume and interactive plots of the MRI images.



