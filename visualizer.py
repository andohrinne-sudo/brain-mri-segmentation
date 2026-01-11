import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st

class MRIVisualizer:
    """
    Visualizes 3D MRI volumes and overlaid tumor segmentation masks.
    """
    def get_labeled_slice(self, image_slice, mask_slice):
        """
        Blends a grayscale MRI slice with colored tumor masks.
        Expects mask_slice in shape (3, H, W).
        """
        # 1. Normalize grayscale anatomical slice (0-255)
        gray_norm = cv2.normalize(image_slice, None, alpha=0, beta=255, 
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 2. Create an RGB base image
        rgb_image = cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2RGB)
        
        # 3. Create the colored mask overlay
        # Channels: 0:Necrotic, 1:Edema, 2:Enhancing
        mask_overlay = np.zeros_like(rgb_image)
        
        # Map prediction channels to RGB colors
        mask_overlay[:, :, 0] = (mask_slice[0, :, :] * 255).astype(np.uint8) # Red
        mask_overlay[:, :, 1] = (mask_slice[1, :, :] * 255).astype(np.uint8) # Green
        mask_overlay[:, :, 2] = (mask_slice[2, :, :] * 255).astype(np.uint8) # Blue
        
        # 4. Blend anatomy with color (70% anatomy, 30% mask)
        return cv2.addWeighted(rgb_image, 0.7, mask_overlay, 0.3, 0)

    def plot_mpr_view(self, image_volume, label_volume, loc=(120, 120, 77)):
        """
        Generates a 3-plane Multi-Planar Reconstruction (MPR) view.
        Corrected to slice (3, H, W, D) labels without out-of-bounds errors.
        """
        # --- Safety Fix: Clip loc to volume boundaries ---
        # BraTS volumes are (240, 240, 155)
        loc = (np.clip(loc[0], 0, 239), np.clip(loc[1], 0, 239), np.clip(loc[2], 0, 154))

        # 1. AXIAL PLANE (Z-axis)
        axial_img = image_volume[:, :, loc[2], 3] 
        # Correct Slicing: Keep all 3 channels for this depth
        axial_mask = label_volume[:, :, :, loc[2]]
        
        # 2. SAGITTAL PLANE (X-axis)
        sagittal_img = image_volume[loc[0], :, :, 3]
        # Correct Slicing: Keep all 3 channels for this width
        sagittal_mask = label_volume[:, loc[0], :, :]
        
        # 3. CORONAL PLANE (Y-axis)
        coronal_img = image_volume[:, loc[1], :, 3]
        # Correct Slicing: Keep all 3 channels for this height
        coronal_mask = label_volume[:, :, loc[1], :]

        # Generate overlays
        view1 = self.get_labeled_slice(axial_img, axial_mask)
        view2 = self.get_labeled_slice(sagittal_img, sagittal_mask)
        view3 = self.get_labeled_slice(coronal_img, coronal_mask)

        # Rendering
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(view1)
        axes[0].set_title(f"Axial (Z={loc[2]})")
        axes[1].imshow(view2)
        axes[1].set_title(f"Sagittal (X={loc[0]})")
        axes[2].imshow(view3)
        axes[2].set_title(f"Coronal (Y={loc[1]})")
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        return fig