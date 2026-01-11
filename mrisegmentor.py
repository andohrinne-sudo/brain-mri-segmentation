import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Activation

class MRISegmentor:
    """
    3D Medical Imaging Diagnostic Engine for Brain Tumor Segmentation.
    Optimized for multi-label (BraTS) tumor regions on low-RAM systems.
    """
    def __init__(self, weights_path="model_pretrained.hdf5", n_labels=3):
        # Set data format to channels_first to match Coursera requirements
        K.set_image_data_format("channels_first")
        self.weights_path = weights_path
        self.n_labels = n_labels
        self.input_shape = (4, 160, 160, 16) 
        
        self.model = self._build_unet()
        
        if self.weights_path:
            try:
                self.model.load_weights(self.weights_path)
                print(f"✅ Diagnostic Engine Loaded: {self.weights_path}")
            except Exception as e:
                print(f"❌ Engine Load Failure: {e}")

    # --- 1. CORE ARCHITECTURE ---
    def _create_conv_block(self, input_layer, n_filters, output_filters):
        """Creates two convolutions; the second one uses 'output_filters'."""
        x = Conv3D(n_filters, (3, 3, 3), padding='same', data_format="channels_first")(input_layer)
        x = Activation('relu')(x)
        x = Conv3D(output_filters, (3, 3, 3), padding='same', data_format="channels_first")(x)
        x = Activation('relu')(x)
        return x

    def _build_unet(self, depth=3, n_base_filters=32):
        """
        Constructs a 15-layer 3D U-Net aligned with pre-trained weights.
        Matches the (3, 3, 3, 32, 64) shape requirement in conv3d_1.
        """
        inputs = Input(self.input_shape)
        current_layer = inputs
        levels = []

        for i in range(depth):
            f1 = n_base_filters * (2**i)
            f2 = f1 * 2  # Doubling filters
            level = self._create_conv_block(current_layer, f1, f2)
            levels.append(level)
            current_layer = MaxPooling3D(pool_size=(2, 2, 2), data_format="channels_first")(level)

        bottleneck_f = n_base_filters * (2**depth)
        current_layer = self._create_conv_block(current_layer, bottleneck_f, bottleneck_f * 2)

        for i in reversed(range(depth)):
            f_dec = n_base_filters * (2**(i+1))
            current_layer = UpSampling3D(size=(2, 2, 2), data_format="channels_first")(current_layer)
            current_layer = concatenate([current_layer, levels[i]], axis=1)
            current_layer = self._create_conv_block(current_layer, f_dec, f_dec)

        output = Conv3D(self.n_labels, (1, 1, 1), activation='sigmoid', data_format="channels_first")(current_layer)
        return Model(inputs=inputs, outputs=output)

    # --- 2. EVALUATION & PREPROCESSING ---
    def preprocess(self, volume):
        """Applies sample-wise zero-mean unit-variance normalization (Float32)."""
        volume = np.nan_to_num(volume).astype(np.float32)
        standardized_vol = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
        return standardized_vol

    @staticmethod
    def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), epsilon=0.00001):
        dice_numerator = 2 * K.sum(y_true * y_pred, axis=axis) + epsilon
        dice_denominator = K.sum(K.square(y_true), axis=axis) + K.sum(K.square(y_pred), axis=axis) + epsilon
        return 1 - K.mean(dice_numerator / dice_denominator)

    def compute_metrics(self, pred, label):
        results = {}
        classes = ['Edema', 'Non-Enhancing', 'Enhancing']
        for i, name in enumerate(classes):
            p, l = pred[i], label[i]
            tp = np.sum((p == 1) & (l == 1))
            tn = np.sum((p == 0) & (l == 0))
            fp = np.sum((p == 1) & (l == 0))
            fn = np.sum((p == 0) & (l == 1))
            results[name] = {
                "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0
            }
        return results

    # --- 3. INFERENCE ENGINE (The part that was missing) ---
    def predict_full_volume(self, image_volume, progress_callback=None):
        """
        Memory-efficient sliding window inference with real-time progress updates.
        :param progress_callback: A function that accepts (current, total) to update the UI.
        """
        full_label = np.zeros([3, 240, 240, 155], dtype=np.float32)
        total_patches = 40 
        patch_count = 0
        
        print(f"🚀 Starting 3D Segmentation (Total Patches: {total_patches})", flush=True)
        
        for x in range(0, 240, 160):
            for y in range(0, 240, 160):
                for z in range(0, 155, 16):
                    patch_count += 1
                    
                    # 1. Update Progress Bar
                    if progress_callback:
                        progress_callback(patch_count, total_patches)
                    
                    # 2. Extract and Process Patch
                    x_end, y_end, z_end = min(x + 160, 240), min(y + 160, 240), min(z + 16, 155)
                    patch = image_volume[x:x_end, y:y_end, z:z_end]
                    
                    if patch_count % 2 == 0 or patch_count == 1:
                        print(f"📦 Progress: {patch_count}/{total_patches} patches...", flush=True)

                    if np.max(patch) > 0:
                        patch = np.nan_to_num(patch).astype(np.float32)
                        std_patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-8)
                        
                        p_data = np.moveaxis(std_patch, 3, 0)
                        model_input = np.zeros([1, 4, 160, 160, 16], dtype=np.float32)
                        model_input[0, :, :p_data.shape[1], :p_data.shape[2], :p_data.shape[3]] = p_data
                        
                        # Prediction
                        pred = self.model.predict(model_input, verbose=0)[0]
                        full_label[:, x:x_end, y:y_end, z:z_end] = pred[:, :x_end-x, :y_end-y, :z_end-z]
        
        print("✅ Full Volume Segmentation Complete.", flush=True)
        return full_label