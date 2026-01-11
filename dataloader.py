import numpy as np
import nibabel as nib
import h5py
import os

class MRIDataLoader:
    """
    Handles ingestion for NiBabel (NifTI) and H5 files.
    Ensures data matches the 15-layer 3D U-Net requirements.
    """
    def __init__(self, patch_dim=(160, 160, 16)):
        self.patch_dim = patch_dim

    def load_case(self, file_path):
        """
        Loads NifTI volumes. Explicitly identifies Nifti1 format to 
        prevent identification errors with temporary files.
        """
        try:
            # Check extension to guide NiBabel's identification
            if file_path.endswith(('.nii', '.nii.gz')):
                img = nib.Nifti1Image.from_filename(file_path)
            else:
                img = nib.load(file_path)
            
            # Use get_fdata() for consistent float conversion
            image = np.array(img.get_fdata())
            return image, None
            
        except Exception as e:
            raise RuntimeError(f"NiBabel Loader Error on {os.path.basename(file_path)}: {e}")

    def load_h5(self, file_path):
        """Loads HDF5 patches (common in BraTS datasets)."""
        with h5py.File(file_path, 'r') as f:
            image = np.array(f.get('x'))
            label = np.array(f.get('y')) if 'y' in f else None
        return image, label

    def get_sub_volume(self, image, label, start_x, start_y, start_z):
        """Generates localized 3D sub-volumes."""
        w, h, d = self.patch_dim
        image_patch = image[start_x: start_x + w, 
                            start_y: start_y + h, 
                            start_z: start_z + d, :]
        
        label_patch = None
        if label is not None:
            label_patch = label[start_x: start_x + w, 
                                start_y: start_y + h, 
                                start_z: start_z + d]
        return image_patch, label_patch

    def standardize(self, patch):
        """Applies zero-mean unit-variance normalization per patch."""
        patch = np.nan_to_num(patch)
        mean = np.mean(patch)
        std = np.std(patch)
        # Prevents division by zero on black background patches
        return (patch - mean) / (std + 1e-8)

    def prepare_for_model(self, patch):
        """
        Formats patch to (1, 4, 160, 160, 16) for 'channels_first'.
        Matches the model's 15-layer depth=3 input requirements.
        """
        # If channels (4) are at the end, move them to position 1
        if patch.shape[-1] == 4:
            patch = np.moveaxis(patch, 3, 0)
        
        # Add batch dimension to satisfy TensorFlow
        return np.expand_dims(patch, axis=0)