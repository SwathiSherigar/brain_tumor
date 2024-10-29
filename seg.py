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

# Define segmentation classes
SEGMENT_CLASSES = {0: 'NOT tumor', 1: 'NECROTIC/CORE', 2: 'EDEMA', 3: 'ENHANCING'}
IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 22

# Define model path
MODEL_PATH = 'model_per_class.h5'

# Dice coefficient functions
def dice_coef(y_true, y_pred, smooth=1.0):
    total_loss = 0
    for i in range(4):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        total_loss += loss
    return total_loss / 4

# Additional metrics functions
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Load the pre-trained model
model = load_model(
    MODEL_PATH,
    custom_objects={"dice_coef": dice_coef, "precision": precision, "sensitivity": sensitivity},
    compile=False
)

# Prediction function
def predict_images(flair_file, t1ce_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_flair:
        tmp_flair.write(flair_file.getvalue())
        flair_path = tmp_flair.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_t1ce:
        tmp_t1ce.write(t1ce_file.getvalue())
        t1ce_path = tmp_t1ce.name

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
    return predictions, flair_path

# # Display function
# def display_predictions(predictions, flair_path, start_slice=70):
#     orig_image = nib.load(flair_path).get_fdata()
#     core, edema, enhancing = predictions[:, :, :, 1], predictions[:, :, :, 2], predictions[:, :, :, 3]

#     fig, axarr = plt.subplots(1, 4, figsize=(20, 10))
#     axarr[0].imshow(cv2.resize(orig_image[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
#     axarr[0].set_title('Original Flair')
    
#     for i, (title, img) in enumerate(zip(['Necrotic/Core', 'Edema', 'Enhancing'], [core, edema, enhancing])):
#         axarr[i + 1].imshow(img[start_slice, :, :], cmap="gray", alpha=0.5)
#         axarr[i + 1].set_title(title)
    
#     st.pyplot(fig)

# # Streamlit Interface
# st.title("MRI Tumor Segmentation")
# flair_file = st.file_uploader("Upload FLAIR Image (.nii)", type="nii")
# t1ce_file = st.file_uploader("Upload T1CE Image (.nii)", type="nii")

# if flair_file and t1ce_file:
#     with st.spinner("Predicting..."):
#         predictions, flair_path = predict_images(flair_file, t1ce_file)
#     display_predictions(predictions, flair_path)


def display_predictions(predictions, flair_path, start_slice=70, gt=None):
    orig_image = nib.load(flair_path).get_fdata()
    core, edema, enhancing = predictions[:, :, :, 1], predictions[:, :, :, 2], predictions[:, :, :, 3]
    
    fig, axarr = plt.subplots(1, 6, figsize=(30, 10))  # Adjust the number of subplots

    # Original image
    axarr[0].imshow(cv2.resize(orig_image[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].set_title('Original Flair')
    
    if gt is not None:  # If ground truth is available
        curr_gt = cv2.resize(gt[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        axarr[1].imshow(curr_gt, cmap="Reds", alpha=0.3)
        axarr[1].set_title('Ground Truth')
    else:
        axarr[1].imshow(np.zeros((IMG_SIZE, IMG_SIZE)), cmap="gray")  # Placeholder if gt is not provided
        axarr[1].set_title('Ground Truth Not Provided')

    # Display all classes (Assuming p contains class predictions)
    all_classes = np.max(predictions, axis=-1)  # Modify this based on how you want to visualize all classes
    axarr[2].imshow(all_classes[start_slice, :, :], cmap="Reds", alpha=1)
    axarr[2].set_title('All Classes')

    # Edema predicted
    axarr[3].imshow(edema[start_slice, :, :], cmap="OrRd", alpha=1)
    axarr[3].set_title(f'{SEGMENT_CLASSES[1]} Predicted')

    # Core predicted
    axarr[4].imshow(core[start_slice, :, :], cmap="OrRd", alpha=1)
    axarr[4].set_title(f'{SEGMENT_CLASSES[2]} Predicted')

    # Enhancing predicted
    axarr[5].imshow(enhancing[start_slice, :, :], cmap="OrRd", alpha=1)
    axarr[5].set_title(f'{SEGMENT_CLASSES[3]} Predicted')

    # Optional: Hide axes
    for ax in axarr:
        ax.axis('off')

    st.pyplot(fig)
flair_file = st.file_uploader("Upload FLAIR Image (.nii)", type="nii")
t1ce_file = st.file_uploader("Upload T1CE Image (.nii)", type="nii")
# Make sure to call display_predictions with the correct parameters
if flair_file and t1ce_file:
    with st.spinner("Predicting..."):
        predictions, flair_path = predict_images(flair_file, t1ce_file)
        gt = None  # Replace with actual ground truth if available
    display_predictions(predictions, flair_path, gt=gt)  # Pass the ground truth if available
