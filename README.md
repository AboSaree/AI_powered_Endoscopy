
# AI-Powered Endoscopy Analysis Tool

## Overview

This application provides automated analysis of endoscopic videos to detect abnormal findings, specifically focusing on Z-line versus esophagitis classification. The system combines deep learning-based frame analysis with YOLO object detection to provide comprehensive diagnostic assistance for medical professionals.

## Features

### Core Functionality
- **Automated Video Analysis**: Process endoscopic videos frame-by-frame using a trained TensorFlow model
- **Abnormal Frame Detection**: Identify frames containing potential abnormalities based on configurable confidence thresholds
- **YOLO Object Detection**: Apply advanced object detection on identified abnormal frames for detailed analysis
- **Real-time Processing**: Live preview of processed frames with visual indicators for abnormal findings
- **Batch Processing**: Efficient processing of multiple frames with customizable skip intervals

### User Interface
- **Intuitive GUI**: PyQt5-based interface with organized control panels
- **Visual Feedback**: Real-time display of processed frames with abnormality indicators
- **Progress Tracking**: Comprehensive progress bars and status updates
- **Results Management**: Automated organization and saving of detected abnormal frames
- **2D Model Viewer**: Integrated viewer for reference model designs

## Technical Requirements

### Dependencies
```
Python 3.7+
tensorflow>=2.0
opencv-python
numpy
PyQt5
ultralytics
```

### Required Files
- `zline_vs_esophagitis_model.h5` - Pre-trained TensorFlow model for abnormality detection
- `best.pt` - YOLO model weights for object detection
- `design.jpg` or `design.png` - Reference 2D model image (optional)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd endoscopy-analysis-tool
```

2. Install required dependencies:
```bash
pip install tensorflow opencv-python numpy PyQt5 ultralytics
```

3. Ensure model files are present:
   - Place `zline_vs_esophagitis_model.h5` in the application directory
   - Place `best.pt` in the application directory
   - Optionally place `design.jpg` or `design.png` for the 2D model viewer

## Usage

### Starting the Application
```bash
python main.py
```

### Workflow

1. **Video Selection**
   - Click "Select Video" to choose an endoscopic video file
   - Supported formats: MP4, AVI, MOV, MKV

2. **Output Configuration**
   - Click "Select Output Directory" to specify where abnormal frames will be saved
   - Click "Set YOLO Output Directory" to specify where YOLO detection results will be saved

3. **Processing Settings**
   - **Abnormal Threshold**: Set the confidence threshold below which frames are considered abnormal (default: 50%)
   - **Frame Skip**: Configure how many frames to skip between analyses (default: 5)
   - **Display Options**: Choose whether to show all processed frames or only abnormal ones

4. **Video Analysis**
   - Click "Start Processing" to begin automated analysis
   - Monitor progress through the progress bar and real-time frame display
   - Abnormal frames are automatically saved to the specified output directory

5. **YOLO Detection**
   - After video processing, click "Run YOLO on Abnormal Frames" to perform detailed object detection
   - Adjust YOLO confidence threshold as needed (default: 30%)
   - Results with detection overlays are saved to the YOLO output directory

6. **Additional Features**
   - Click "2D Model" to view reference design images
   - Monitor processing statistics and completion summaries

## Architecture

### Core Components

**VideoProcessor Class**
- Handles video file processing and frame analysis
- Implements multi-threading for responsive UI during processing
- Manages model predictions and threshold-based classification

**YoloProcessor Class**
- Performs YOLO object detection on previously identified abnormal frames
- Supports batch processing of image collections
- Generates annotated output images with detection overlays

**MainWindow Class**
- Provides the primary user interface
- Coordinates between different processing components
- Manages file I/O and user interactions

**DesignViewer Class**
- Displays reference 2D model images
- Supports multiple image formats and automatic path detection

### Processing Pipeline

1. **Video Input**: Load and validate endoscopic video files
2. **Frame Extraction**: Extract frames at specified intervals
3. **Preprocessing**: Resize and normalize frames for model input
4. **Classification**: Apply TensorFlow model for abnormality detection
5. **Filtering**: Save frames below the abnormality threshold
6. **Object Detection**: Apply YOLO detection to abnormal frames
7. **Output Generation**: Save annotated results and processing statistics

## Model Requirements

### TensorFlow Model Specifications
- **Input Shape**: (224, 224, 3) - RGB images resized to 224x224 pixels
- **Output**: Single probability value (0-1 range)
- **Interpretation**: Values below threshold indicate abnormal findings

### YOLO Model Specifications
- **Format**: YOLOv8 compatible weights file (.pt)
- **Input**: Variable size images (default: 640x640)
- **Output**: Bounding boxes with confidence scores and class predictions

## Output Structure

```
output_directory/
├── abnormal_frames/
│   ├── abnormal_frame_0001.jpg
│   ├── abnormal_frame_0002.jpg
│   └── ...
└── yolo_output/
    ├── abnormal_frame_0001_detected.jpg
    ├── abnormal_frame_0002_detected.jpg
    └── ...
```

## Configuration Options

### Processing Parameters
- **Abnormal Threshold**: 1-99% (default: 50%)
- **Frame Skip Interval**: 1-30 frames (default: 5)
- **YOLO Confidence**: 10-90% (default: 30%)

### Display Options
- Real-time frame preview with abnormality indicators
- Configurable display of all frames vs. abnormal frames only
- Visual markers for detected abnormalities

## Performance Considerations

- **Memory Usage**: Proportional to video resolution and batch size
- **Processing Speed**: Dependent on hardware capabilities and frame skip settings
- **Storage Requirements**: Abnormal frames are saved as individual JPEG files

## Troubleshooting

### Common Issues

**Model Loading Errors**
- Ensure `zline_vs_esophagitis_model.h5` is present and accessible
- Verify TensorFlow installation and compatibility

**Video Processing Failures**
- Check video file format compatibility
- Ensure adequate disk space in output directory
- Verify video file is not corrupted

**YOLO Detection Issues**
- Confirm `best.pt` file is present and valid
- Check ultralytics package installation
- Ensure sufficient abnormal frames exist for processing

### Error Messages
- Check console output for detailed error information
- Verify all file paths and permissions
- Ensure all dependencies are properly installed

## Team Members

This project was developed by:

- **Muhammad Nasser**
- **Abdelrahman Emad**
- **Farah Yehia**
- **Alaa Essam**
- **Amat Elrahman**

## Contributing

This tool is designed for medical research and diagnostic assistance. When contributing:

- Follow medical software development best practices
- Ensure thorough testing of any modifications
- Maintain compatibility with existing model formats
- Document any changes to processing algorithms

## License

This project is intended for research and educational purposes. Please ensure compliance with relevant medical device regulations and data privacy requirements when using with patient data.

## Support

For technical issues or questions about the implementation, please refer to the documentation or submit an issue through the repository's issue tracking system.
