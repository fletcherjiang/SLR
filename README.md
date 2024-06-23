# Sign Language Recognition System README

## Overview
This system interprets 24 specific hand gestures representing letters A-Y (excluding J and Z) using labels 1-24. It aims to efficiently translate these gestures into text using machine learning models. Considerations for the system's development include data collection, feature extraction, model training, and accuracy improvement.



## Demonstration Video

![Demo Video](https://github.com/fletcherjiang/SLR/tree/main/videoA1_video.mp4)

## Models
### Supported Models
- **Support Vector Machine (SVC)**

- **K-Nearest Neighbors (KNN)**

  

### Model Selection
Experiments were performed with SVC and KNN on real-world datasets, whereas CNN was evaluated using simulation data only.

## Dataset
- **Training**: Sign Language MNIST for initial training. Performance was suboptimal in practical applications, leading to the creation of an original dataset capturing 24 gestures from various angles.
- **Real-world Testing**: Conducted with custom datasets to better mimic practical usage.

## Installation Guide
Ensure Python and necessary libraries (TensorFlow, NumPy) are installed. For real-time gesture recognition, Google's MediaPipe Hands API is utilized.

### Dependencies
- Python 3.x
- TensorFlow
- NumPy
- OpenCV (for image processing enhancements)
- MediaPipe

### Setup
1. Clone the repository: `git clone https://github.com/fletcherjiang/SLR`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the system: `python NewsignLanguageRecognition.py`

## Features and Testing
### Feature Extraction
- **Image-based**: Utilizes 28x28 pixel grayscale images.
- **Hand Joint-based**: Employs the coordinates of 21 hand joints for more robust feature extraction.

### Testing
Tests were carried out with both types of feature extraction methods. Joint-based features showed superior real-world performance and robustness against variable backgrounds and hand sizes.

## Issues and Improvements
### Background Interference
- **Problem**: Inconsistencies in gesture recognition across different backgrounds.
- **Solution**: Implemented a preprocessing algorithm that standardizes the background using the YCrCb color space and contour extraction techniques to isolate the hand gesture.

### Palm Size Variation
- **Problem**: Variability in recognition accuracy across different hand sizes.
- **Solution**: Enriching the dataset with a variety of hand sizes improved model training and reduced recognition errors.

## Conclusion
The SVC model combined with nodal point detection for feature extraction proved to be the most effective. The system achieves over 95% accuracy in varied test scenarios, demonstrating robustness across diverse environments and user demographics.

## Additional Notes
To use the MediaPipe API, ensure proper installation as per the guidelines provided. This API enhances the system's stability and recognition capabilities.

---

