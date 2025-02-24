import SimpleITK as sitk
import h5py
import numpy as np
import scipy.io
from visualize_with_slider import visualize_volume_with_slider
import tifffile 

def Load_from_HDF5(file_path=None, file_format= 'hdf5'):
    if file_format == 'hdf5':
        # read hdf5
        # Replace 'your_file.h5' with the path to your HDF5 file
        # file_path = r'E:\LoDoPaB\ground_truth_train\ground_truth_train_000.hdf5'
        if file_path is None:
            file_path = r'D:\Data\LoDoPaB\ground_truth_train\ground_truth_train_000.hdf5'
        # Open the HDF5 file and load the dataset
        with h5py.File(file_path, 'r') as f:
            dataset = f['data'][:]
    elif file_format == 'dicom':
        # read dicom 
        if file_path is None:
            patient_folder = r"D:\Data\dataNeaotomAlpha\24010916\54100000"
        else:
            patient_folder = file_path
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(patient_folder)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        # Added a call to PermuteAxes to change the axes of the data
        #image = sitk.PermuteAxes(image, [2, 1, 0])
        dataset = sitk.GetArrayFromImage(image)
    print('data shape:', dataset.shape)
    return dataset

def load_sino_volume_from_mat(file_path):
    mat_data = scipy.io.loadmat(file_path)
    # Extract the matrix
    data = mat_data['data_Th1']
    angles = mat_data['Angle'].squeeze()
    angles = angles / 180 * np.pi
    print(data.shape)
    print(angles.shape)
    print("angular range:", angles[0], angles[-1])
    return data, angles

def load_interpolated_sino_volume_from_mat(file_path):
    print('load data from', file_path)
    mat_data = scipy.io.loadmat(file_path)
    # Extract the matrix
    data_with_interp = mat_data['sino_with_interpolation']
    data_without_interp = mat_data['sino_without_interpolation']
    z_interpolated_detector_positions = mat_data['z_interpolated_detector_positions'].squeeze()
    x_interpolated_detector_positions = mat_data['x_interpolated_detector_positions'].squeeze()
    TablePosition_mov_ref = mat_data['TablePosition_mov_ref'].squeeze()
    angles = mat_data['Angle_aligned'].squeeze()
    angles = angles / 180 * np.pi
    print(data_with_interp.shape)
    print(angles.shape)
    print("angular range:", angles[0], angles[-1])
    print("z_interpolated_detector_positions range:", z_interpolated_detector_positions[0], z_interpolated_detector_positions[-1])
    print("x_interpolated_detector_positions range:", x_interpolated_detector_positions[0], x_interpolated_detector_positions[-1])
    print("TablePosition_mov range:", TablePosition_mov_ref[0], TablePosition_mov_ref[-1])
    return data_with_interp,data_without_interp, angles,z_interpolated_detector_positions,x_interpolated_detector_positions,TablePosition_mov_ref

def load_sino_slice_from_mat(file_path):
    mat_data = scipy.io.loadmat(file_path)

    # Extract the matrix
    sino_with_calibration = mat_data['sino_with_calibration']#.transpose(2, 1, 0)
    sino_without_calibration = mat_data['sino_without_calibration']#.transpose(2, 1, 0)
    angles = mat_data['Angle'].squeeze()
    print('sino_with_calibration', sino_with_calibration.shape, sino_with_calibration.dtype)
    print("angular range:", angles[0], angles[-1])
    angles = angles / 180 * np.pi
    return sino_with_calibration, sino_without_calibration, angles

def load_processed_sino_from_tif(file_path):
    image_data = tifffile.imread(file_path)

if __name__=='__main__':
    # G:\Data\LoDoPaB\ground_truth_train\ground_truth_train_000.hdf5
    mode = "Load_from_HDF5"
    if mode == "Load_from_HDF5":
        #dataset = Load_from_HDF5(file_path=r'G:\data\LoDoPaB\observation_train\observation_train_000.hdf5', file_format= 'hdf5')
        dataset = Load_from_HDF5(file_path=r'G:\data\LoDoPaB\ground_truth_train\ground_truth_train_000.hdf5', file_format= 'hdf5')
    elif mode == "load_interpolated_sino_volume_from_mat":
        data_with_interp,data_without_interp, angles,z_interpolated_detector_positions,x_interpolated_detector_positions,TablePosition_mov_ref = load_interpolated_sino_volume_from_mat(r"G:\projects\ct_data_process\matlab\Naeotom_readraw\output\sino_interpolated_Th1_3000_2016.mat")
        dataset = data_with_interp
    visualize_volume_with_slider(dataset, 0)

