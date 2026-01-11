import streamlit as st
import numpy as np
import os
import tempfile
import cv2
from mrisegmentor import MRISegmentor
from dataloader import MRIDataLoader
from visualizer import MRIVisualizer

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="AI Brain MRI Segmentor", layout="wide")
st.title("🧠 3D Brain Tumor Segmentation Engine")

# --- 2. INITIALIZE COMPONENTS ---
@st.cache_resource
def load_engine():
    # Loads the 15-layer 3D U-Net engine
    return MRISegmentor(weights_path="model_pretrained.hdf5")

engine = load_engine()
loader = MRIDataLoader()
viz = MRIVisualizer()

# --- 3. SIDEBAR & NAVIGATION ---
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a file (.nii, .nii.gz, or .h5)", type=["nii", "gz", "h5"])
    
    st.subheader("Plane Navigation")
    # Coordinates are clipped to 240x240x155 to avoid segfaults
    slice_idx = st.slider("Axial (Z)", 0, 154, 75)
    sag_idx = st.slider("Sagittal (X)", 0, 239, 120)
    cor_idx = st.slider("Coronal (Y)", 0, 239, 120)

# --- 4. PROCESSING PIPELINE ---
if uploaded_file is not None:
    original_name = uploaded_file.name
    file_ext = original_name.split('.')[-1]
    
    # Save uploaded file to a temporary location to save persistent /home space
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # File Loading
        if file_ext == 'h5':
            with st.spinner("Loading H5 data..."):
                image_vol, _ = loader.load_h5(tmp_path)
        else:
            with st.spinner("Reading NifTI volume..."):
                image_vol, _ = loader.load_case(tmp_path)

        # AI Segmentation Logic with Progress Bar
        if st.button("🚀 Run AI Segmentation"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total):
                pct = current / total
                progress_bar.progress(pct)
                status_text.text(f"Analyzing 3D patch {current} of {total}...")

            # Run patch-based inference using the verified engine
            prediction = engine.predict_full_volume(image_vol, progress_callback=update_progress)
            st.session_state['pred'] = prediction
            st.success("Analysis Complete!")

        # --- 5. RESULTS VISUALIZATION ---
        st.divider()
        if 'pred' in st.session_state:
            # Generate the 3-plane MPR view
            fig = viz.plot_mpr_view(image_vol, st.session_state['pred'], 
                                   loc=(sag_idx, cor_idx, slice_idx))
            st.pyplot(fig)
            
            # Detailed Side-by-Side View
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Anatomical View")
                # Normalize to 0-255 uint8 to prevent "Data outside [0, 1]" error
                ax_img = image_vol[:, :, slice_idx, 3]
                norm_img = cv2.normalize(ax_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                st.image(norm_img, use_container_width=True) # Updated parameter name
            with col2:
                st.subheader("AI Tumor Map")
                # Overlay showing Necrotic, Edema, and Enhancing regions
                overlay = viz.get_labeled_slice(ax_img, st.session_state['pred'][:, :, :, slice_idx])
                st.image(overlay, use_container_width=True)
        else:
            st.info("Upload a scan and click 'Run' to begin.")

    except Exception as e:
        st.error(f"⚠️ Application Error: {e}")
    finally:
        # Prevent disk overflow by deleting temp files immediately
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
else:
    st.warning("Please upload an MRI file to begin.")

# --- 6. FOOTER ---