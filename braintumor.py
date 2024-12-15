import streamlit as st
import keras.backend as K
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import cv2
from keras.models import load_model
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mayavi import mlab
import configobj
import imageio
import nilearn.plotting as nlplt
import nilearn as nl
from io import BytesIO
import gif_your_nifti.core as gif2nif
from scipy.ndimage import label
# Define segmentation classes
SEGMENT_CLASSES = {0: 'NOT tumor', 1: 'NECROTIC/CORE', 2: 'EDEMA', 3: 'ENHANCING'}
IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 22

# Define model path
MODEL_PATH = 'model_per_class.h5'
# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
   #     K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
#    K.print_tensor(total_loss, message=' total dice coef: ')
    return total_loss

def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)



# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    
# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

model = keras.models.load_model('model_per_class.h5', 
                                   custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                   "dice_coef": dice_coef,
                                                   "precision": precision,
                                                   "sensitivity":sensitivity,
                                                   "specificity":specificity,
                                                   "dice_coef_necrotic": dice_coef_necrotic,
                                                   "dice_coef_edema": dice_coef_edema,
                                                   "dice_coef_enhancing": dice_coef_enhancing
                                                  }, compile=False)

# Prediction function
def predict_images(flair_file, t1ce_file,seg_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_flair:
        tmp_flair.write(flair_file.getvalue())
        flair_path = tmp_flair.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_t1ce:
        tmp_t1ce.write(t1ce_file.getvalue())
        t1ce_path = tmp_t1ce.name
        
    with tempfile.NamedTemporaryFile(delete=None, suffix=".nii") as tmp_seg:
        tmp_seg.write(seg_file.getvalue())
        seg_path = tmp_seg.name

    # Load the images
    flair = nib.load(flair_path).get_fdata()
    t1ce = nib.load(t1ce_path).get_fdata()
    
    # Prepare data for prediction
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    for i in range(VOLUME_SLICES):
        X[i, :, :, 0] = cv2.resize(flair[:, :, i + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        X[i, :, :, 1] = cv2.resize(t1ce[:, :, i + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
    
    # Get predictions
    predictions = model.predict(X / np.max(X), verbose=1)
    return predictions, flair_path, seg_path


def display_predictions(predictions, flair_path,seg_path, start_slice=70, gt=None):
    orig_image = nib.load(flair_path).get_fdata()
    core, edema, enhancing = predictions[:, :, :, 1], predictions[:, :, :, 2], predictions[:, :, :, 3]
    
    fig, axarr = plt.subplots(1, 6, figsize=(30, 10))  # Adjust the number of subplots

    # Original image
    axarr[0].imshow(cv2.resize(orig_image[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].set_title('Original Flair')
    
    # if gt is not None:  # If ground truth is available
    #     curr_gt = cv2.resize(gt[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    #     axarr[1].imshow(curr_gt, cmap="Reds", alpha=0.3)
    #     axarr[1].set_title('Ground Truth')
    # else:
    #     axarr[1].imshow(np.zeros((IMG_SIZE, IMG_SIZE)), cmap="gray")  # Placeholder if gt is not provided
    #     axarr[1].set_title('Ground Truth Not Provided')
    for i in range(6): # for each image, add brain background
        axarr[i].imshow(cv2.resize(orig_image[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
          
    gt = nib.load(seg_path).get_fdata()

    axarr[0].imshow(cv2.resize(orig_image[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].title.set_text('Original image flair')
    curr_gt=cv2.resize(gt[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST)
    axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3) # ,alpha=0.3,cmap='Reds'
    axarr[1].title.set_text('Ground truth')
    axarr[2].imshow(predictions[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[2].title.set_text('all classes')
    axarr[3].imshow(edema[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    axarr[4].imshow(core[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[5].imshow(enhancing[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')

    # Optional: Hide axes
    for ax in axarr:
        ax.axis('off')

    st.pyplot(fig)


def calculate_biomarkers(predictions, flair_path):
    flair_img = nib.load(flair_path)
    voxel_dims = flair_img.header.get_zooms()
    voxel_volume = np.prod(voxel_dims)  # Volume of a single voxel in mm³

    tumor_volume = {}
    
    for i, class_label in SEGMENT_CLASSES.items():
        if i != 0:  # Skip the background
            class_mask = predictions[..., i] > 0.5  # Create binary mask for class
            labeled_array, num_features = label(class_mask)  # Label connected components

            voxel_count = np.sum(class_mask)
            tumor_count = 0

            if num_features > 0:
                # Loop through each labeled component
                for j in range(1, num_features + 1):
                    tumor_mask = (labeled_array == j)  # Create mask for the j-th tumor
                    tumor_voxel_count = np.sum(tumor_mask)

                    # Set a threshold to filter out small tumors (optional)
                    if tumor_voxel_count > 100:  # Adjust this threshold as needed
                        tumor_count += 1
                
                tumor_volume[class_label] = {
                    "voxel_count": voxel_count,
                    "volume_mm3": voxel_count * voxel_volume,
                    "tumor_count": tumor_count
                }
            else:
                # If no tumor found, handle accordingly
                tumor_volume[class_label] = {
                    "voxel_count": 0,
                    "volume_mm3": 0.0,
                    "tumor_count": 0
                }

    return tumor_volume



def display_predictions_with_biomarkers(predictions, flair_path):
    biomarkers = calculate_biomarkers(predictions, flair_path)
    st.write("### Tumor Biomarker Analysis")
    
    for label, volume_info in biomarkers.items():
        st.write(f"{label} Volume: {volume_info['voxel_count']} voxels, "
                 f"{volume_info['volume_mm3']:.2f} mm³ "
               )
    st.write( f"Tumor Count: {volume_info['tumor_count']}")


def metastatic(predictions, flair_path):
#     # Calculate biomarkers for all tumor classes
   
    biomarkers = calculate_biomarkers(predictions, flair_path)
    st.write("### Meta Ststaic Analysis")
    for label,volume_info in biomarkers.items():    
    # Determine metastasis based on volume threshold
        if volume_info['volume_mm3']> 1000:
            st.write(f"**{label}:** Metastasis detected")
        else:
            st.write(f"**{label}:** No metastasis detected")
    



def visualize_t1_t2(t1_data, t2_data):
    """
    Visualize T1 and T2 weighted images.
    
    Parameters:
    - t1_data (ndarray): T1-weighted image data.
    - t2_data (ndarray): T2-weighted image data.
    """
    slice_index = t1_data.shape[2] // 2

    plt.figure(figsize=(12, 6))

    # T1 image
    plt.subplot(1, 2, 1)
    plt.imshow(t1_data[:, :, slice_index], cmap='gray')
    plt.title('T1-weighted Image')
    plt.axis('off')

    # T2 image
    plt.subplot(1, 2, 2)
    plt.imshow(t2_data[:, :, slice_index], cmap='gray')
    plt.title('T2-weighted Image')
    plt.axis('off')

    st.pyplot(plt)

def create_gif_from_nifti(nifti_file, fps=5):
    # Load the NIfTI file using nibabel
    img = nib.load(nifti_file)
    data = img.get_fdata()  # Get the 3D volume data
    
    # Normalize data for visualization
    data = (255 * (data - np.min(data)) / (np.max(data) - np.min(data))).astype(np.uint8)

    # Collect slices along one axis (e.g., axial)
    slices = [data[:, :, i] for i in range(data.shape[2])]
    
    # Save slices as a GIF
    gif_bytes = imageio.mimwrite('<bytes>', slices, format='GIF', fps=fps)

    return gif_bytes

def plot_nifti_images(flair_file, seg_file):
    """
    Plot NIfTI images using nilearn and matplotlib, and return a BytesIO object with the plot.
    
    Parameters:
    - flair_file (str): Path to the FLAIR NIfTI image file.
    - seg_file (str): Path to the segmentation mask NIfTI file.
    
    Returns:
    - BytesIO: The in-memory plot image.
    """
    # Load the FLAIR image and segmentation mask
    niimg = nl.image.load_img(flair_file)
    nimask = nl.image.load_img(seg_file)
    
    # Create subplots
    fig, axes = plt.subplots(nrows=4, figsize=(10, 15))
    
    # Plot various representations of the FLAIR image
    nlplt.plot_anat(niimg, title='FLAIR Image - plot_anat', axes=axes[0])
    nlplt.plot_epi(niimg, title='FLAIR Image - plot_epi', axes=axes[1])
    nlplt.plot_img(niimg, title='FLAIR Image - plot_img', axes=axes[2])
    nlplt.plot_roi(nimask, title='FLAIR Image with Mask - plot_roi', bg_img=niimg, axes=axes[3], cmap='Paired')
    
    # Adjust spacing between subplots
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # Save plot to an in-memory file with tight bounding box
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", bbox_inches="tight")
    img_buffer.seek(0)
    plt.close(fig)  # Close the plot to free memory
    
    return img_buffer







flair_file = st.file_uploader("Upload FLAIR Image (.nii)", type="nii")
t1ce_file = st.file_uploader("Upload T1CE Image (.nii)", type="nii")
seg_file = st.file_uploader("Upload Segmentation Image (.nii)", type="nii")
# Make sure to call display_predictions with the correct parameters

   
t1_file = st.file_uploader("Upload T1 Image (.nii)", type="nii")
t2_file = st.file_uploader("Upload T2 Image (.nii)", type="nii")

if t1_file is not None and t2_file is not None and flair_file is not None and seg_file is not None:
    # Save uploaded files to temporary files
    try:
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as t1_temp_file:
            t1_temp_file.write(t1_file.read())
            t1_temp_file_name = t1_temp_file.name
        
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as t2_temp_file:
            t2_temp_file.write(t2_file.read())
            t2_temp_file_name = t2_temp_file.name
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as flair_temp_file:
            flair_temp_file.write(flair_file.read())
            flair_temp_file_name = flair_temp_file.name
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as temp_seg:
            temp_seg.write(seg_file.read())
                    
      
        # Load the temporary files using nibabel
        t1_img = nib.load(t1_temp_file_name)
        t2_img = nib.load(t2_temp_file_name)
        flair_img=nib.load(flair_temp_file_name)
        # Get the image data
        t1_data = t1_img.get_fdata()
        t2_data = t2_img.get_fdata()
        if flair_file and t1ce_file and seg_file:
           with st.spinner("Predicting..."):
                predictions, flair_path, seg_path = predict_images(flair_file, t1ce_file,seg_file)
      
                display_predictions(predictions, flair_path,seg_path)  # Pass the ground truth if available
                display_predictions_with_biomarkers(predictions, flair_path)
                metastatic(predictions, flair_path )
        # Visualize the images
        st.write("### Diffusion Weighted Images")
        visualize_t1_t2(t1_data, t2_data)
          # 3D Volume Visualization
      
        gif_data = create_gif_from_nifti(flair_temp_file_name)
        st.image(gif_data, caption="3d Volumetric analysis", use_container_width=True, output_format="GIF")
        plot_image = plot_nifti_images(flair_temp_file.name, temp_seg.name)
        st.image(plot_image, caption="FLAIR and Mask Visualization", use_container_width=True)

   

        # Display the plot
    except Exception as e:
        st.error(f"An error occurred: {e}")
