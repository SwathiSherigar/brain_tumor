import numpy as np
import cv2
import nibabel as nib
import keras
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

# Load the Keras model
model = keras.models.load_model('model_per_class.h5', compile=False)

# Constants
IMG_SIZE = 128  # Example value, adjust as needed
SEGMENT_CLASSES = ['Background', 'Edema', 'Non-enhancing tumor', 'Enhancing tumor']  # Example classes

# Function to load an MRI image
def imageLoader(path):
    image = nib.load(path).get_fdata()
    return image  # Return the raw data without resizing

# Function to normalize image data
def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)  # Normalize to [0, 1]

# Function to show predictions by ID
def showPredictsById(flair_path, seg_path, start_slice=70):
    gt = nib.load(seg_path).get_fdata()
    origImage = imageLoader(flair_path)

    # Simulate predictions (replace this with actual model predictions)
    p = np.random.rand(1, IMG_SIZE, IMG_SIZE, 4)  # Replace with actual prediction logic

    # Get predictions for core, edema, and enhancing tumor
    core = p[0, :, :, 1]
    edema = p[0, :, :, 2]
    enhancing = p[0, :, :, 3]

    # Apply thresholding to create binary masks for better visualization
    core_binary = (core > 0.3).astype(np.float32)
    edema_binary = (edema > 0.3).astype(np.float32)
    enhancing_binary = (enhancing > 0.3).astype(np.float32)

    # Resize ground truth for consistent display
    gt_resized = cv2.resize(gt[:, :, start_slice], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # Normalize and prepare the original image
    origImage_normalized = normalize_image(origImage[:, :, start_slice])
    origImage_resized = cv2.resize(origImage_normalized, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # Create a figure for displaying images
    plt.figure(figsize=(20, 10))
    f, axarr = plt.subplots(1, 6, figsize=(20, 10))

    # Set background color to dark
    f.patch.set_facecolor('black')
    
    # Original Image
    axarr[0].imshow(origImage_resized, cmap="gray")
    axarr[0].title.set_text('Original image (Flair)')
    axarr[0].axis('off')  # Hide axes
    axarr[0].set_facecolor('white')

    # Ground Truth
    axarr[1].imshow(gt_resized, cmap="Reds", interpolation='none', alpha=0.3)
    axarr[1].title.set_text('Ground Truth')
    axarr[1].axis('off')
    axarr[1].set_facecolor('black')

    # All Classes Overlay
    axarr[2].imshow(origImage_resized, cmap="gray", interpolation='none')
    axarr[2].imshow(p[0, :, :, 1:4].max(axis=2), cmap="Reds", interpolation='none', alpha=0.3)
    axarr[2].title.set_text('All Classes')
    axarr[2].axis('off')
    axarr[2].set_facecolor('black')

    # Edema Prediction as solid color
    axarr[3].imshow(origImage_resized, cmap="gray", interpolation='none')
    axarr[3].imshow(edema_binary, cmap="OrRd", interpolation='none', alpha=0.3)  # Solid color
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} Predicted')
    axarr[3].axis('off')
    axarr[3].set_facecolor('black')

    # Core Prediction as solid color
    axarr[4].imshow(origImage_resized, cmap="gray", interpolation='none')
    axarr[4].imshow(core_binary, cmap="OrRd", interpolation='none', alpha=0.3)  # Solid color
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} Predicted')
    axarr[4].axis('off')
    axarr[4].set_facecolor('black')

    # Enhancing Prediction as solid color
    axarr[5].imshow(origImage_resized, cmap="gray", interpolation='none')
    axarr[5].imshow(enhancing_binary, cmap="OrRd", interpolation='none', alpha=0.3)  # Solid color
    axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} Predicted')
    axarr[5].axis('off')
    axarr[5].set_facecolor('black')

    # Convert the figure to BGR format for OpenCV
    plt.savefig('temp.png', bbox_inches='tight', dpi=300)
    img = cv2.imread('temp.png')
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR

    # Save the figure instead of displaying it
    output_path = 'prediction_results.png'  # Specify your output path
    cv2.imwrite(output_path, img_bgr)  # Save as BGR image
    plt.close()  # Close the figure to free memory

    # Inform the user that the figure has been saved
    messagebox.showinfo("Info", f"Prediction results saved to {output_path}")

# GUI setup
class MRIApp:
    def __init__(self, master):
        self.master = master
        master.title("MRI Prediction GUI")

        self.label_flair = tk.Label(master, text="Select Flair MRI Image:")
        self.label_flair.pack()

        self.flair_button = tk.Button(master, text="Select Flair Image", command=self.load_flair_image)
        self.flair_button.pack()

        self.label_seg = tk.Label(master, text="Select Segmentation Image:")
        self.label_seg.pack()

        self.seg_button = tk.Button(master, text="Select Segmentation Image", command=self.load_seg_image)
        self.seg_button.pack()

        self.start_slice_label = tk.Label(master, text="Enter Start Slice (0):")
        self.start_slice_label.pack()

        self.start_slice_entry = tk.Entry(master)
        self.start_slice_entry.insert(0, "0")
        self.start_slice_entry.pack()

        self.predict_button = tk.Button(master, text="Show Predictions", command=self.show_predictions)
        self.predict_button.pack()

    def load_flair_image(self):
        self.flair_path = filedialog.askopenfilename(filetypes=[("NIfTI Files", "*.nii")])
        if not self.flair_path:
            messagebox.showwarning("Warning", "No file selected for Flair image.")

    def load_seg_image(self):
        self.seg_path = filedialog.askopenfilename(filetypes=[("NIfTI Files", "*.nii")])
        if not self.seg_path:
            messagebox.showwarning("Warning", "No file selected for Segmentation image.")

    def show_predictions(self):
        start_slice = int(self.start_slice_entry.get())
        try:
            showPredictsById(self.flair_path, self.seg_path, start_slice)
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Run the GUI
root = tk.Tk()
app = MRIApp(root)
root.mainloop()
