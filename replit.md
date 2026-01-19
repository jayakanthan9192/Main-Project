# SMS Spam Detection System

## Overview

This is a comprehensive SMS spam detection system built with Streamlit that uses machine learning to identify spam content across multiple formats (text, images, and videos). The system employs a multi-modal approach combining GAN-based text classification, CNN-based image analysis, OCR text extraction, and video frame processing to detect spam across different media types.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Purpose**: Provides an interactive web interface for spam detection across multiple input types
- **Key Features**:
  - Text message spam detection
  - Image-based spam detection with OCR support
  - Video spam detection through frame extraction
  - Sample data demonstration
  - Real-time predictions with confidence scores

### Machine Learning Models

#### GAN-based Text Spam Detector
- **Architecture**: Generative Adversarial Network with discriminator for classification
- **Components**:
  - Generator: Creates synthetic spam/ham messages
  - Discriminator: Classifies messages as spam or legitimate
- **Text Processing**:
  - Custom vocabulary builder (5000 words)
  - Sequence padding to 100 tokens
  - 128-dimensional embeddings
  - 100-dimensional latent space
- **Rationale**: GANs can learn complex patterns in spam messages and generate realistic training data

#### CNN-based Image Spam Detector
- **Architecture**: Convolutional Neural Network with multiple conv blocks
- **Structure**:
  - Three convolutional blocks with batch normalization
  - Progressive channel expansion (32 → 64 → 128)
  - Max pooling and dropout for regularization
  - Dense layers for classification
- **Input**: 224x224 RGB images
- **Rationale**: CNNs excel at extracting visual features from images, useful for detecting promotional graphics and spam imagery

### Text Extraction Pipeline

#### OCR Processor
- **Engine**: Pytesseract (Tesseract OCR wrapper)
- **Preprocessing Steps**:
  - Grayscale conversion
  - OTSU thresholding for binarization
  - Fast non-local means denoising
- **Quality Control**:
  - Confidence threshold filtering (30% minimum)
  - Average confidence score calculation
- **Rationale**: OCR enables text-based spam detection on image/video content, expanding detection capabilities beyond pure visual analysis

#### Video Processor
- **Technology**: OpenCV (cv2)
- **Frame Extraction Strategy**:
  - Evenly distributed frame sampling
  - Configurable intervals (default: every 30th frame)
  - Maximum frame limit (10 frames) to manage processing load
- **Output**: PIL Image objects for downstream processing
- **Rationale**: Videos can contain spam in text overlays or visual content; frame extraction allows reuse of image detection models

### Data Flow

1. **Text Input**: Text → GAN Text Detector → Classification
2. **Image Input**: Image → (OCR → Text Detector) + Image CNN → Combined Classification
3. **Video Input**: Video → Frame Extraction → Process each frame as image → Aggregated Results

### Sample Data System
- **Purpose**: Demonstration and testing
- **Content**: Curated spam and legitimate message examples
- **Categories**: 
  - Spam samples (promotional, phishing, scams)
  - Legitimate samples (personal, business communications)

## External Dependencies

### Machine Learning Frameworks
- **TensorFlow/Keras**: Core deep learning framework for GAN and CNN models
- **NumPy**: Numerical computations and array operations

### Computer Vision & OCR
- **OpenCV (cv2)**: Video processing and image preprocessing
- **Pytesseract**: OCR text extraction from images
- **Pillow (PIL)**: Image manipulation and format conversion

### Web Framework
- **Streamlit**: Interactive web application framework
- **Matplotlib/Seaborn**: Data visualization and results presentation

### System Dependencies
- **Tesseract OCR**: External binary required by pytesseract (must be installed on system)

### Python Standard Library
- **tempfile**: Temporary file handling for uploads
- **os**: File system operations
- **re**: Regular expression processing for text cleaning
- **pickle**: Model serialization (referenced but not shown in implementation)