import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
def visualize_volume_with_slider(volume, slice_dimension=0):
    """
    Visualize 3D volume with a slider to manually control the displayed slice.
    
    Args:
        volume (numpy.ndarray): The 3D volume data.
    """
    if volume is None:
        print("No volume data found.")
        return
    
    slices = volume.shape[slice_dimension]  # Assuming slices are along the 3rd dimension

    # Create a figure and axis for visualization
    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(left=0.25, bottom=0.25)  # Adjust space for the slider
    
    # Display the first slice initially
    slice_idx = 0
    if slice_dimension==0:
        slice = volume[slice_idx,:, :] 
    elif slice_dimension==1:
        slice = volume[:,slice_idx, :] 
    elif slice_dimension==2:
        slice = volume[:, :, slice_idx] 

    img_display = ax.imshow(slice, cmap='gray')
    ax.set_title(f"Slice {slice_idx + 1}/{slices}")
    
    # Create a slider for slice selection
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slice_slider = Slider(ax_slider, 'Slice', 0, slices - 1, valinit=slice_idx, valstep=1)
    
    # Update the displayed slice when the slider is moved
    def update(val):
        slice_idx = int(slice_slider.val)
        if slice_dimension==0:
            slice = volume[slice_idx,:, :] 
        elif slice_dimension==1:
            slice = volume[:,slice_idx, :] 
        elif slice_dimension==2:
            slice = volume[:, :, slice_idx] 
        img_display.set_data(slice)
        ax.set_title(f"Slice {slice_idx + 1}/{slices}")
        fig.canvas.draw_idle()  # Update the plot

    # Attach the update function to the slider
    slice_slider.on_changed(update)
    
    plt.show()
