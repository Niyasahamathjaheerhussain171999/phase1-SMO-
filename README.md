# Football Video Analysis - SMO Project (Phase 1)

A comprehensive football video analysis system using YOLO (You Only Look Once) for player detection, pass tracking, and performance analysis. This project analyzes football match videos to detect passes, calculate accuracy metrics, and provide detailed statistics.

## 🎯 Project Overview

This project implements an AI-powered football analysis system that:
- **Detects players** in football videos using YOLO object detection
- **Tracks passes** between players (short and long passes)
- **Calculates accuracy metrics** (overall accuracy, pass success rates)
- **Generates detailed statistics** and visualizations
- **Validates results** with image-based validation system

## 📁 Project Structure

```
SMO good accuracy/
├── setup/                    # Setup and installation scripts
│   ├── colab_cell_1_setup.py      # GPU setup and check
│   ├── colab_cell_2_install.py    # Dependencies installation
│   ├── colab_cell_3_drive.py      # Google Drive mount (Colab)
│   └── colab_cell_4_upload.py     # Video file upload
│
├── src/
│   ├── classes/              # Football analysis classes
│   │   ├── colab_cell_5_class.py           # Base analysis class
│   │   ├── colab_cell_5_class_A_PLUS.py    # A+ grade optimized class
│   │   └── colab_cell_5_class_BALANCED.py  # Balanced configuration class
│   │
│   ├── analysis/             # Analysis execution scripts
│   │   ├── colab_cell_6_analysis.py        # Base analysis with YouTube
│   │   ├── colab_cell_6_analysis_A_PLUS.py # A+ grade analysis
│   │   └── colab_cell_6_analysis_FIXED.py # Fixed analysis version
│   │
│   └── validation/           # Validation scripts
│       ├── colab_cell_7_validation.py      # Video validation
│       └── colab_cell_7_image_validation.py # Image-based validation
│
├── docs/                     # Documentation
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🚀 Features

### Core Functionality
- **Player Detection**: Uses YOLO (YOLOv8) for real-time player detection
- **Pass Detection**: Automatically detects passes between players
- **Pass Classification**: Categorizes passes as "short" or "long"
- **Success/Failure Tracking**: Determines pass success rates
- **Accuracy Metrics**: Calculates overall, short, and long pass accuracy
- **Performance Analysis**: Tracks pass patterns and statistics

### Analysis Classes
1. **FixedFootballAnalysis** (Base): Standard analysis with balanced settings
2. **APlusFootballAnalysis** (A+): Optimized for 95%+ accuracy with strict filtering
3. **BalancedFootballAnalysis**: Balanced configuration for moderate accuracy

### Key Metrics Calculated
- Total passes detected
- Short vs Long pass distribution
- Success/failure rates
- Overall accuracy percentage
- Pass distance distribution
- Confidence scores

## 📋 Requirements

### Python Dependencies
See `requirements.txt` for full list. Key dependencies:
- `ultralytics` - YOLO object detection
- `opencv-python` - Video processing
- `pandas` - Data analysis
- `numpy` - Numerical operations
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - Machine learning metrics
- `torch`, `torchvision` - PyTorch (for GPU support)
- `yt-dlp` - YouTube video download (optional)
- `easyocr` - OCR capabilities (optional)

### Hardware Requirements
- **GPU Recommended**: CUDA-enabled GPU for faster processing
- **Memory**: At least 8GB RAM
- **Storage**: Sufficient space for video files and results

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Niyasahamathjaheerhussain171999/phase1-SMO-.git
   cd phase1-SMO-
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For Google Colab**: Upload the notebook cells in order (Cell 1 → Cell 7)

## 📖 Usage

### Google Colab Workflow

1. **Setup (Cell 1)**: Check GPU availability
   ```python
   !nvidia-smi
   ```

2. **Install (Cell 2)**: Install all dependencies
   ```python
   !pip install ultralytics opencv-python ...
   ```

3. **Mount Drive (Cell 3)**: Optional - Mount Google Drive
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Upload Video (Cell 4)**: Upload your football video
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

5. **Initialize Class (Cell 5)**: Choose your analysis class
   - `FixedFootballAnalysis` - Standard
   - `APlusFootballAnalysis` - High accuracy (95%+)
   - `BalancedFootballAnalysis` - Balanced settings

6. **Run Analysis (Cell 6)**: Execute analysis on your video
   ```python
   passes, accuracy = analyzer.analyze_video(video_file)
   ```

7. **Validate Results (Cell 7)**: Validate AI predictions
   ```python
   validation_report = create_image_validation_system()
   ```

### Local Usage

For local execution, adapt the scripts to remove Colab-specific code:
- Remove `!` commands (convert to regular Python)
- Replace Google Colab file uploads with local file paths
- Adjust paths for your system

## 📊 Output

The analysis generates:
- **CSV files**: Detailed pass data (`detailed_football_results.csv`)
- **JSON files**: Accuracy metrics (`detailed_accuracy_metrics.json`)
- **Visualizations**: Multiple charts and graphs
- **Validation images**: Sample frames with AI predictions
- **Validation reports**: Comprehensive validation data

## 🎯 Analysis Results Example

```
🏆 DETAILED FOOTBALL ANALYSIS RESULTS
============================================================
📊 TOTAL PASSES: 347
🎯 OVERALL ACCURACY: 82.3%
⏱️ PROCESSING TIME: 15.2 minutes

📈 PASS TYPE BREAKDOWN:
🔵 SHORT PASSES:
   Total: 234
   ✅ Success: 198 (84.6%)
   ❌ Failure: 36 (15.4%)
   🎯 Accuracy: 85.2%

🔴 LONG PASSES:
   Total: 113
   ✅ Success: 87 (77.0%)
   ❌ Failure: 26 (23.0%)
   🎯 Accuracy: 77.0%
```

## 🔧 Configuration Options

### Analysis Classes Comparison

| Class | Confidence Threshold | Pass Distance Range | Accuracy Target | Use Case |
|-------|---------------------|---------------------|----------------|----------|
| **FixedFootballAnalysis** | 0.3 | 30-300 pixels | 80-85% | Standard analysis |
| **APlusFootballAnalysis** | 0.6 | 50-200 pixels | 95%+ | High accuracy needed |
| **BalancedFootballAnalysis** | 0.4 | 40-250 pixels | 85-90% | Balanced performance |

## 📝 Validation

The project includes two validation methods:

1. **Video Validation**: Real-time frame-by-frame validation
2. **Image Validation** (Recommended): Creates timestamped images for manual review

Validation images include:
- Frame timestamps
- Player IDs
- Pass distance
- Pass type (short/long)
- Success/failure status
- Confidence scores

## 🤝 Contributing

This is Phase 1 of the SMO project. Future improvements may include:
- Real ball tracking
- Team identification
- Player movement patterns
- Advanced statistics
- Frontend integration

## 📄 License

[Specify your license here]

## 👤 Author

**Niyasahamathjaheerhussain171999**

## 🙏 Acknowledgments

- YOLO/Ultralytics for object detection
- OpenCV for video processing
- Google Colab for computational resources

## 📞 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project was originally designed for Google Colab. For local execution, adapt the code to remove Colab-specific dependencies.

