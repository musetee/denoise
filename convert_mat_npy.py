import scipy.io
import numpy as np
import json
import os
def convert_mat_to_npy(mat_file_path, output_dir):
    # Load the .mat file
    mat_filename = mat_file_path  # Change this to your .mat file name
    data = scipy.io.loadmat(mat_filename)

    # Extract the necessary variables
    angles = data['Angle'].flatten().tolist()  # Convert to a list
    projections = data['data_Th1']  # Shape: (1376, 144, 1008)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Save each projection as a separate .npy file
    for i in range(projections.shape[2]):
        npy_filename = os.path.join(output_dir, f"projection_{i:04d}.npy")
        np.save(npy_filename, projections[:, :, i])

    # Save angles as a JSON meta file
    json_filename = os.path.join(output_dir, "angles.json")
    with open(json_filename, "w") as json_file:
        json.dump({"angles": angles}, json_file, indent=4)

    print(f"Saved {projections.shape[2]} .npy files and angles.json in '{output_dir}' directory.")


# Example usage
mat_path = r"G:\projects\ct_data_process\matlab\Naeotom_readraw\output\output_file_Th1_3000_1008.mat"
output_dir = r"data\output_file_Th1_3000_1008"
convert_mat_to_npy(mat_path, output_dir)
