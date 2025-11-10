# Streamlit Frontend for Lung Cancer Prediction

## Quick Start

### Running the Streamlit App

1. **Make sure the model is trained** (if not already done):
   ```bash
   python "Lung Cancer Prediction.py"
   ```
   This will create `trained_lung_cancer_model.h5` file.

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** - Streamlit will automatically open a browser window, or you can navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## Features

- ✅ **Simple Upload Interface**: Drag and drop or click to upload lung CT scan images
- ✅ **Real-time Prediction**: Get instant predictions with confidence scores
- ✅ **Visual Results**: See prediction probabilities in a bar chart
- ✅ **Detailed Analysis**: View probabilities for all classes
- ✅ **No Training Required**: Uses the pre-trained model (training is already complete!)

## Important Notes

### Training Status
- **Training is COMPLETE** ✅
- The model was trained when you ran `Lung Cancer Prediction.py`
- The app **does NOT retrain** the model - it only loads the saved model for predictions
- Training only happens when you run the training script explicitly

### Model Files
- `trained_lung_cancer_model.h5` - Full model (used by the app)
- `best_model.weights.h5` - Best weights checkpoint (used during training)

### Supported Image Formats
- PNG
- JPG/JPEG

### Image Requirements
- Images are automatically resized to 350x350 pixels
- RGB color images are expected
- Images are normalized (pixel values divided by 255) before prediction

## Troubleshooting

### Model Not Found Error
If you see "Model not found", make sure:
1. You've run the training script first
2. The `trained_lung_cancer_model.h5` file exists in the project directory

### Import Errors
If you get import errors, install required packages:
```bash
pip install streamlit pillow tensorflow
```

## Usage

1. Start the app: `streamlit run app.py`
2. Upload an image using the file uploader
3. View the prediction results:
   - Predicted class name
   - Confidence percentage
   - Probability bar chart
   - Detailed probabilities for all classes

## Model Information

- **Architecture**: Xception (Transfer Learning)
- **Input Size**: 350x350 pixels
- **Classes**: 4 (Normal, Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma)
- **Training Accuracy**: ~78.8%
- **Validation Accuracy**: ~62.5%

