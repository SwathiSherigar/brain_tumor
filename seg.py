import streamlit as st
import keras.backend as K
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import cv2

# DEFINE seg-areas  
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3 later
}
IMG_SIZE = 128
VOLUME_SLICES = 100 
VOLUME_START_AT = 22  # first slice of volume that we will include

# Dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:, :, :, i])
        y_pred_f = K.flatten(y_pred[:, :, :, i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss

# Define per class evaluation of dice coef
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 1] * y_pred[:, :, :, 1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 1])) + K.sum(K.square(y_pred[:, :, :, 1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 2] * y_pred[:, :, :, 2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 2])) + K.sum(K.square(y_pred[:, :, :, 2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 3] * y_pred[:, :, :, 3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 3])) + K.sum(K.square(y_pred[:, :, :, 3])) + epsilon)

# Computing Precision 
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# Load model
model = keras.models.load_model('model_per_class.h5', 
                                 custom_objects={ 
                                     'accuracy': tf.keras.metrics.MeanIoU(num_classes=4),
                                     "dice_coef": dice_coef,
                                     "precision": precision,
                                     "sensitivity": sensitivity,
                                     "specificity": specificity,
                                     "dice_coef_necrotic": dice_coef_necrotic,
                                     "dice_coef_edema": dice_coef_edema,
                                     "dice_coef_enhancing": dice_coef_enhancing
                                 }, compile=False)

def predictByPath(flair_path, t1ce_path):
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    
    flair = nib.load(flair_path).get_fdata()
    ce = nib.load(t1ce_path).get_fdata() 

    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        
    return model.predict(X / np.max(X), verbose=1)
def showPredictsById(seg_path, flair_path, t1ce_path, start_slice=70):
    gt = nib.load(seg_path).get_fdata()
    origImage = nib.load(flair_path).get_fdata()
    p = predictByPath(flair_path, t1ce_path)

    core = p[:, :, :, 1]
    edema = p[:, :, :, 2]
    enhancing = p[:, :, :, 3]

    fig, axarr = plt.subplots(1, 6, figsize=(20, 50))
    axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].title.set_text('Original image flair')
    curr_gt=cv2.resize(gt[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST)
    axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3) # ,alpha=0.3,cmap='Reds'
    axarr[1].title.set_text('Ground truth')
    axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=1)
    axarr[2].title.set_text('all classes')
    axarr[3].imshow(edema[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=1)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    axarr[4].imshow(core[start_slice,:,], cmap="OrRd", interpolation='none', alpha=1)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[5].imshow(enhancing[start_slice,:,], cmap="OrRd", interpolation='none', alpha=1)
    axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')

    # Show the plot in Streamlit
    st.pyplot(fig)

# Streamlit GUI
st.title("Brain Tumor Segmentation")
st.write("Upload FLAIR and T1CE images for prediction.")

# File uploader for images
flair_file = st.file_uploader("Upload FLAIR Image (.nii)", type=["nii"])
t1ce_file = st.file_uploader("Upload T1CE Image (.nii)", type=["nii"])

if flair_file and t1ce_file:
    # Read images from uploaded files
    flair_path = flair_file.name
    t1ce_path = t1ce_file.name
    with open(flair_path, "wb") as f:
        f.write(flair_file.getbuffer())
    with open(t1ce_path, "wb") as f:
        f.write(t1ce_file.getbuffer())

    if st.button("Predict"):
        showPredictsById('BraTS20_Training_001_seg.nii', flair_path, t1ce_path)

st.write("### Instructions:")
st.write("1. Upload the FLAIR and T1CE images.")
st.write("2. Click on 'Predict' to view the segmentation results.")
