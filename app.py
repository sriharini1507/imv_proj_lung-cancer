import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="wide"
)

# Title and description
st.title("🫁 Lung Cancer Prediction System")
st.markdown("Upload a lung CT scan image to predict the type of lung cancer or if it's normal.")

# Constants
IMAGE_SIZE = (350, 350)
MODEL_PATH = 'trained_lung_cancer_model.h5'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
train_folder = os.path.join(DATASET_DIR, 'train')

@st.cache_data
def get_class_labels():
    """Get class labels from training folder structure"""
    if os.path.exists(train_folder):
        # Get sorted class folders
        class_folders = sorted([d for d in os.listdir(train_folder) 
                               if os.path.isdir(os.path.join(train_folder, d))])
        return class_folders
    else:
        # Fallback to default labels if folder doesn't exist
        return [
            'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
            'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
            'normal',
            'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
        ]

def get_class_display_name(class_label):
    """Get simplified display name for a class label"""
    if 'adenocarcinoma' in class_label.lower():
        return 'Adenocarcinoma'
    elif 'large.cell' in class_label.lower() or 'large_cell' in class_label.lower():
        return 'Large Cell Carcinoma'
    elif 'squamous' in class_label.lower():
        return 'Squamous Cell Carcinoma'
    elif 'normal' in class_label.lower():
        return 'Normal'
    else:
        return class_label

# Get class labels dynamically
CLASS_LABELS = get_class_labels()

@st.cache_resource
def load_trained_model():
    """Load the trained model (cached to avoid reloading)"""
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            return model
        else:
            st.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(img, target_size):
    """Preprocess image for prediction"""
    # Convert RGBA to RGB if necessary (remove alpha channel)
    if img.mode == 'RGBA':
        # Create a white background
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = rgb_img
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize(target_size)
    # Convert to array
    img_array = image.img_to_array(img)
    # Expand dimensions to match model input
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values (same as training)
    img_array /= 255.0
    return img_array

def predict_image(model, img_array):
    """Make prediction on preprocessed image"""
    predictions = model.predict(img_array, verbose=0)
    return predictions[0]

# Load model
model = load_trained_model()

if model is not None:
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a lung CT scan image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a PNG, JPG, or JPEG image of a lung CT scan"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Uploaded Image")
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width='stretch')
        
        with col2:
            st.subheader("🔍 Prediction Results")
            
            # Preprocess image
            img_array = preprocess_image(img, IMAGE_SIZE)
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                predictions = predict_image(model, img_array)
            
            # Get predicted class
            predicted_idx = np.argmax(predictions)
            predicted_class = CLASS_LABELS[predicted_idx]
            confidence = predictions[predicted_idx] * 100
            
            # Display prediction
            st.success(f"**Predicted Class:** {get_class_display_name(predicted_class)}")
            st.info(f"**Confidence:** {confidence:.2f}%")
            
            # Display all class probabilities
            st.subheader("📊 Prediction Probabilities")
            prob_data = {
                'Class': [get_class_display_name(label) for label in CLASS_LABELS],
                'Probability (%)': [prob * 100 for prob in predictions]
            }
            
            # Create a bar chart
            import pandas as pd
            df = pd.DataFrame(prob_data)
            df = df.sort_values('Probability (%)', ascending=False)
            st.bar_chart(df.set_index('Class'))
            
            # Show detailed probabilities
            with st.expander("View Detailed Probabilities"):
                for i, (label, prob) in enumerate(zip(CLASS_LABELS, predictions)):
                    prob_percent = prob * 100
                    display_name = get_class_display_name(label)
                    # Highlight the predicted class
                    if i == predicted_idx:
                        st.markdown(f"**{display_name}:** {prob_percent:.2f}% ✅")
                    else:
                        st.markdown(f"{display_name}: {prob_percent:.2f}%")
    
    else:
        st.info("👆 Please upload an image to get started")
        
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a deep learning model based on Xception architecture 
        to classify lung CT scan images into four categories:
        
        - **Normal**: No cancer detected
        - **Adenocarcinoma**: A type of lung cancer
        - **Large Cell Carcinoma**: A type of lung cancer
        - **Squamous Cell Carcinoma**: A type of lung cancer
        
        **Note:** This is a research tool and should not be used as a substitute 
        for professional medical diagnosis.
        """)
        
        st.header("📋 Model Information")
        if model is not None:
            st.success("✅ Model loaded successfully")
            st.caption(f"Model: Xception-based CNN")
            st.caption(f"Input size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels")
        else:
            st.error("❌ Model not loaded")
        
        st.header("🔧 Instructions")
        st.markdown("""
        1. Click on the file uploader above
        2. Select a lung CT scan image (PNG, JPG, or JPEG)
        3. Wait for the prediction results
        4. View the confidence scores for each class
        """)

else:
    st.error("""
    **Model not found!**
    
    Please run the training script first:
    ```bash
    python "Lung Cancer Prediction.py"
    ```
    
    This will train the model and save it as `trained_lung_cancer_model.h5`
    """)

