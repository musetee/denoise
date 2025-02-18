import numpy as np
import matplotlib.pyplot as plt
import cv2

def convert_png_npy(png_filename, npy_filename=None):
    """
    Loads a PNG image and saves it as an NPY file.

    Parameters:
        png_filename (str): Path to the PNG image.
        npy_filename (str): Path to save the NPY file (default: same name as PNG).
    """
    # Set default NPY filename if not provided
    if npy_filename is None:
        npy_filename = png_filename.replace(".png", ".npy")

    # Load the PNG image as a NumPy array (grayscale or color)
    image_array = cv2.imread(png_filename, cv2.IMREAD_UNCHANGED)  # Preserve transparency if available
    print("image shape: ", image_array.shape)
    # Convert BGR to RGB if the image has color channels
    if len(image_array.shape) == 3 and image_array.shape[-1] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # Save the NumPy array
    np.save(npy_filename, image_array)
    print(f"Saved image as {npy_filename}")

# Example Usage:
# save_png_as_npy("example.png")  # Saves as "example.npy"
