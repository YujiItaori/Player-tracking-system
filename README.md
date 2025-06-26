# Player Tracking System using YOLOv11 and DeepSORT

## 📖 Overview

This project implements a real-time player tracking system for sports videos using YOLOv11 for object detection and DeepSORT for multi-object tracking. The system can detect and track multiple players across video frames, assigning unique IDs to each player and maintaining their identity throughout the video sequence.

![Screenshot 2025-06-26 122614](https://github.com/user-attachments/assets/a37defe3-9a48-48e1-9232-727778705d8c)
![Screenshot 2025-06-26 122614](https://github.com/user-attachments/assets/a37defe3-9a48-48e1-9232-727778705d8c)



## 🎯 Features

- **Real-time Player Detection**: Uses YOLOv11 model for accurate player detection
- **Multi-Object Tracking**: Implements DeepSORT algorithm for robust player tracking
- **Unique ID Assignment**: Each detected player gets a unique tracking ID
- **Visual Feedback**: Displays bounding boxes, player IDs, and tracking information
- **CPU Optimized**: Designed to run efficiently on CPU-only systems
- **Video Processing**: Processes MP4 video files and outputs tracked results

## 🔧 Dependencies and Environment Requirements

### Python Version
- Python 3.8 or higher

### Required Libraries
Install the required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Key Dependencies
- **ultralytics==8.3.159**: YOLOv11 implementation
- **deep-sort-realtime==1.3.2**: DeepSORT tracking algorithm
- **opencv-python==4.11.0.86**: Computer vision operations
- **torch==2.7.1**: PyTorch for deep learning
- **numpy==2.3.1**: Numerical computations
- **scipy==1.16.0**: Scientific computing

## 📁 Project Structure

```
player-tracking-system/
├── app.py                 # Main tracking script
├── requirements.txt        # Python dependencies
├── base.pt                # YOLOv11 model file
├── 15sec_input_720p.mp4   # Input video file
├── output_video.mp4       # Output tracked video
└── README.md              # This file
```

## 🚀 How to Set Up and Run the Code

### Step 1: Clone or Download the Project
```bash
git clone  https://github.com/YujiItaori/Player-tracking-system.git
cd Player-tracking-system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Required Files
Ensure you have the following files in your project directory:
- `base.pt` - Your fine-tuned YOLOv11 model and you can find it here https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view
- `input_video.mp4` - Input video file

### Step 4: Run the Tracking System
```bash
python app.py
```

### Step 5: View Results
The processed video with tracking information will be saved as `output_video.mp4`

## 🎮 Usage

### Basic Usage
```python
python app.py
```

### Configuration Options
You can modify the following parameters in the code:

#### Detection Parameters
```python
# In the inference section
results = model(frame, conf=0.25, iou=0.5, verbose=False)
```
- `conf`: Confidence threshold (0.0-1.0)
- `iou`: IoU threshold for NMS (0.0-1.0)

#### Tracking Parameters
```python
tracker = DeepSort(
    max_iou_distance=0.7,    # Maximum IoU distance for matching
    max_age=30,              # Maximum frames to keep lost tracks
    n_init=3,                # Frames needed to confirm a track
    nn_budget=100,           # Maximum features stored per class
    max_cosine_distance=0.4  # Maximum cosine distance for matching
)
```

## 📊 Output Information

The system provides real-time feedback showing:
- **Frame Count**: Current frame being processed
- **Detections**: Number of players detected in current frame
- **Tracked**: Number of players being actively tracked
- **Player IDs**: Unique identifier for each tracked player
- **Bounding Boxes**: Visual representation of player locations

## 🔧 Customization

### Changing Input/Output Files
```python
input_path = "input_video.mp4"
output_path = "output_video.mp4"
```

### Adjusting Detection Classes
The system automatically detects "player" or "person" classes. To use different classes:
```python
# Modify the class detection logic
player_class_ids = [0, 1, 2]  # Your specific class IDs
```

### Video Display (Optional)
To see real-time processing, uncomment these lines:
```python
cv2.imshow('Tracking', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

## 🛠️ System Requirements

### Minimum Requirements
- **CPU**: Multi-core processor (Intel i5 or AMD equivalent)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space for dependencies and models
- **OS**: Windows 10/11, macOS 10.14+, or Linux Ubuntu 18.04+

### Recommended Specifications
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **RAM**: 16GB or higher
- **Storage**: SSD for faster model loading

## 📈 Performance Notes

- **CPU Processing**: Optimized for CPU-only systems
- **Memory Usage**: Approximately 2-4GB RAM during processing
- **Processing Speed**: Varies based on video resolution and CPU performance
- **Accuracy**: Depends on model quality and video conditions

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```
   Solution: Ensure 'base.pt' file is in the project directory
   ```

2. **Video File Not Found**
   ```
   Solution: Check that input video file exists and path is correct
   ```

3. **Memory Issues**
   ```
   Solution: Reduce video resolution or process shorter video segments
   ```

4. **No Players Detected**
   ```
   Solution: Adjust confidence threshold or check model compatibility
   ```

## 📝 License

This project is for educational and research purposes. Please ensure you have appropriate licenses for any commercial use.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure all dependencies are correctly installed
4. Verify input files are accessible

## 🔄 Version History

- **v1.0**: Initial implementation with YOLOv11 and DeepSORT
- **v1.1**: Added coordinate validation and error handling
- **v1.2**: Improved tracking parameters and visual feedback

---

**Note**: This system is designed for sports video analysis and player tracking applications. Performance may vary based on video quality, lighting conditions, and player visibility.
