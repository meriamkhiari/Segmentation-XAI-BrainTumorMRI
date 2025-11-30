# Brain Tumor Segmentation with Explainable AI

A comprehensive web application for brain tumor segmentation using deep learning, featuring explainable AI (XAI) capabilities through Grad-CAM visualization and LLM-powered textual explanations.

## 🎯 Features

- **Deep Learning Segmentation**: U-Net based model for accurate brain tumor segmentation
- **Explainable AI (XAI)**: 
  - Grad-CAM heatmaps for visual attention visualization
  - LLM-powered textual explanations using LLaMA-3
  - Attention analysis and alignment metrics
- **Web Interface**: Flask-based user-friendly web application
- **Comprehensive Metrics**: Tumor area, centroid, bounding box, and contour analysis
- **Real-time Processing**: Fast inference with GPU support (CUDA)
- **Export Capabilities**: JSON export of analysis results

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Technologies](#technologies)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster inference)
- Ollama (for LLaMA-3 LLM features)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Brain-Tumor-Segmentation
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Ollama (Optional, for LLM features)

If you want to use the LLM-powered explanations:

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull the LLaMA-3 model:
   ```bash
   ollama pull llama3:latest
   ```

### Step 5: Download Model Weights

Place your trained model weights (`best_model.pth`) in the `models/` directory.

## 💻 Usage

### Starting the Application

```bash
python app.py
```

The application will start on `http://localhost:5000` by default.

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Upload a brain MRI image (PNG, JPG, or JPEG format)
3. Wait for processing (segmentation + XAI analysis)
4. View results:
   - Segmentation mask visualization
   - Grad-CAM heatmap
   - Overlay visualization
   - Tumor metrics
   - LLM-generated explanation
5. Download the analysis results as JSON

### Programmatic Usage

```python
from data.segmentation_model import BrainTumorSegmentationModel
from agents.xai_agent import XAIAgent
from utils.image_processing import load_and_preprocess_image
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BrainTumorSegmentationModel("models/best_model.pth", device=device)

# Preprocess image
image_tensor, original_img = load_and_preprocess_image("path/to/image.png", device)

# Get segmentation
prediction_mask = model.predict(image_tensor)

# Get XAI explanation
xai_agent = XAIAgent(model, image_tensor, prediction_mask.cpu().numpy())
xai_results = xai_agent.run()
```

## 📁 Project Structure

```
Brain-Tumor-Segmentation/
├── agents/                    # Agent modules
│   ├── segmentation_agent.py # Segmentation orchestration agent
│   └── xai_agent.py          # Explainable AI agent
├── data/                      # Data and model definitions
│   └── segmentation_model.py # U-Net model implementation
├── models/                    # Trained model weights
│   └── best_model.pth        # Model checkpoint
├── outputs/                   # Analysis results (JSON files)
├── templates/                 # Flask HTML templates
│   ├── index.html            # Main upload page
│   └── results.html          # Results display page
├── temp_uploads/             # Temporary uploaded images
├── utils/                     # Utility functions
│   ├── gradcam.py            # Grad-CAM implementation
│   ├── image_processing.py   # Image preprocessing utilities
│   └── llama3_llm.py         # LLaMA-3 LLM wrapper
├── app.py                     # Flask application entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file (optional) for configuration:

```env
# LLM Configuration (Optional)
LLAMA_API_KEY=your_api_key_here
LLAMA_MODEL=llama3:latest

# Flask Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True

# Model Configuration
MODEL_PATH=models/best_model.pth
DEVICE=cuda  # or 'cpu'
```

### Model Configuration

The segmentation model uses:
- **Architecture**: U-Net with ResNet-34 encoder
- **Input**: 3-channel RGB images (256x256)
- **Output**: Binary segmentation mask
- **Loss Function**: Dice Loss

## 🌐 API Endpoints

### `GET /`
- Main page for image upload

### `POST /`
- Upload MRI image for analysis
- Returns: Results page with segmentation and XAI visualizations

### `GET /download/<filename>`
- Download analysis JSON file
- Example: `/download/outputs/analysis_20251125_161013.json`

## 🛠️ Technologies

- **Deep Learning**: PyTorch, torchvision
- **Segmentation**: segmentation-models-pytorch
- **Web Framework**: Flask
- **Image Processing**: OpenCV, Pillow, Albumentations
- **Explainability**: Grad-CAM, LLaMA-3 (via Ollama)
- **Visualization**: Matplotlib
- **LLM Integration**: LangChain, Ollama

## 🧠 Model Architecture

The segmentation model is based on U-Net architecture:

- **Encoder**: ResNet-34 (pretrained weights not used)
- **Decoder**: U-Net decoder with skip connections
- **Input Channels**: 3 (RGB)
- **Output Channels**: 1 (binary segmentation)
- **Activation**: Sigmoid for probability output

### Training Details

- **Loss Function**: Dice Loss with smoothing factor (1e-6)
- **Optimization**: Adam optimizer (typical)
- **Data Augmentation**: Albumentations pipeline

## 📊 Output Format

The application generates JSON files with the following structure:

```json
{
  "timestamp": "2025-11-25T16:10:13",
  "image_filename": "brain_scan.png",
  "tumor_analysis": {
    "area_pixels": 1234,
    "area_percentage": 1.88,
    "centroid": {
      "x": 128,
      "y": 128
    },
    "bounding_box": {
      "x": 50,
      "y": 50,
      "width": 100,
      "height": 100
    },
    "contour_count": 1
  },
  "xai_explanation": {
    "textual_justification": "The model identified a tumor region..."
  }
}
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use CPU mode
   - Set `device="cpu"` in the code

2. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Verify model is pulled: `ollama list`
   - The application will continue without LLM features if Ollama is unavailable

3. **Model Loading Error**
   - Verify model file exists at `models/best_model.pth`
   - Check model architecture matches the code

4. **Image Processing Errors**
   - Ensure images are in supported formats (PNG, JPG, JPEG)
   - Check image dimensions (will be resized to 256x256)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- U-Net architecture: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Grad-CAM: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
- segmentation-models-pytorch library
- Ollama for local LLM inference

## 📧 Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: This application is for research and educational purposes. Always consult medical professionals for actual medical diagnosis.

