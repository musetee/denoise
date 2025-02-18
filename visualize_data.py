import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from matplotlib.widgets import Slider
import os
def visualize_npy_slices(folder_path):
    """
    Visualizes .npy slices from a given folder using a slider.
    
    Parameters:
        folder_path (str): Path to the folder containing .npy files.
    """
    # Get list of .npy files sorted numerically
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    
    if not npy_files:
        print("No .npy files found in the given directory.")
        return
    
    file_path0 = os.path.join(folder_path, npy_files[0])
    first_slice = np.load(file_path0)
    first_slice = np.rot90(first_slice, k=1, axes=(0,1))

    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(left=0.25, bottom=0.25)  # Adjust space for the slider
    img_display = ax.imshow(first_slice, cmap='gray')
    ax.set_title(f"Slice")
    # Create a slider for slice selection
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slice_slider = Slider(ax_slider, 'Slice', 0, len(npy_files)-1, valinit=0, valstep=1)
    
    def update(val):
        index = int(slice_slider.val)
        """Loads and displays a slice based on the slider index."""
        file_path = os.path.join(folder_path, npy_files[index])
        image_data = np.load(file_path)
        image_data = np.rot90(image_data, k=1, axes=(0,1))
        img_display.set_data(image_data)
        ax.set_title(f"Slice {index + 1}/{len(npy_files)}")
        fig.canvas.draw_idle()  # Update the plot
    
    slice_slider.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    test_folder = r"G:\projects\selfsupdenoise\data\output_file_Th1_3000_1008"
    visualize_npy_slices(test_folder)