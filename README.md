# 🤖 ASL (American Sign Language) Recognition System

A comprehensive machine learning-based ASL recognition system using flex sensors and MPU6050 IMU connected to an ESP32 microcontroller. Features advanced data collection, real-time visualization, calibration, and multiple ML/DL models for accurate gesture recognition.

## 📊 System Overview

**Reference Flowchart**: `asl_system_flowchart.png` - Complete visual workflow from hardware to real-time prediction

The system processes information through 7 interconnected stages:
```
Hardware → Firmware → Data Collection → Preprocessing → ML Training → Prediction → Output
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Hardware Setup
- Connect flex sensors and MPU6050 to ESP32 as per wiring diagram. **Note:** If using a 35-pin ESP32 board, the pinky flex sensor (FLEX5) is connected to GPIO 4 in the provided `esp32_asl.ino` firmware (originally GPIO 36). Please verify the pinout for your specific board.
- Upload `esp32_asl/esp32_asl.ino` to your ESP32
- Connect ESP32 to your computer via USB

### 3. Data Collection (Choose Method)

#### Method 1: Advanced Data Collection (Recommended)
```bash
python asl_data_collector.py
```
**Features:**
- Real-time sensor visualization
- Automatic calibration
- Data validation and quality control
- Interactive menu system
- Progress tracking and statistics

#### Method 2: Simple Data Collection
```bash
python asl_train.py
```
**Features:**
- Basic data collection
- Manual gesture recording
- CSV export with timestamps

### 4. Data Preprocessing
```bash
python data_preprocessing.py
```
**Process:**
- Load and combine all data sources
- Scale features and encode labels
- Generate data visualizations
- Save preprocessed data for ML training

### 5. Train and Evaluate AI/ML Models

#### Classic Machine Learning
```bash
python asl_ml_model.py
```
**Models:** Random Forest, SVM, Gradient Boosting, Neural Network
**Features:** Cross-validation, performance comparison, model saving

#### Deep Learning (TensorFlow/Keras)
```bash
python asl_deep_learning.py
```
**Models:** Deep Neural Network with multiple layers
**Features:** Advanced architecture, detailed evaluation, visualization

### 6. Real-time AI Prediction
```bash
python pc_asl_tts.py
```
**Features:**
- Real-time gesture recognition
- Text-to-speech output
- Confidence threshold filtering
- Live sensor data processing

## 🔧 Enhanced System Features

### **Advanced Data Collection System**
- **Sensor Calibration**: Automatic calibration for accurate readings
- **Real-time Visualization**: Live plots of flex sensors and IMU data
- **Data Validation**: Quality control and stability checking
- **Multi-threading**: Smooth UI with background data collection
- **Configuration Management**: Centralized settings in `data_collection_config.py`

### **Improved ESP32 Firmware**
- **Stable Readings**: Multiple samples with variance checking
- **Command Interface**: Send commands via serial (CALIBRATE, STATUS, DATA)
- **Offset Compensation**: Remove sensor bias for accuracy
- **Error Handling**: Detect unstable readings and hardware issues

### **Comprehensive Documentation**
- **Information Feeding Guide**: Step-by-step instructions for each stage
- **Data Collection Guide**: Detailed setup and usage instructions
- **System Flowchart**: Visual representation of entire workflow
- **Troubleshooting Section**: Common issues and solutions

## 📊 Data Sources & Management

### **Available Data Sources**
1. **Real Sensor Data**: Collected from your ESP32 glove
2. **Synthetic Data**: Generated realistic sensor patterns
3. **External Datasets**: Public ASL datasets
4. **Preprocessed Data**: ML-ready arrays and scalers

### **Data Quality Control**
- **Validation Rules**: Check sensor ranges and stability
- **Calibration**: Automatic sensor calibration
- **Visualization**: Real-time and post-processing plots
- **Statistics**: Sample counts, distributions, and quality metrics

## 🤖 Machine Learning & Deep Learning

### **Classic ML Models**
| Model | Features | Best Use Case |
|-------|----------|---------------|
| Random Forest | Ensemble, robust | General purpose |
| SVM | High accuracy | Small datasets |
| Gradient Boosting | Sequential learning | Complex patterns |
| Neural Network | Non-linear patterns | Feature relationships |

### **Deep Learning**
- **Architecture**: Multi-layer neural network
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout and early stopping
- **Evaluation**: Comprehensive metrics and visualizations

### **Model Performance**
- **Accuracy**: >90% with proper data
- **Real-time**: <500ms prediction latency
- **Reliability**: >99% uptime during operation

## 📁 Enhanced Project Structure

```
ssipmt asl/
├── 📁 esp32_asl/
│   └── esp32_asl.ino              # Enhanced ESP32 firmware
├── 📁 asl_datasets/               # External datasets
├── 📁 preprocessed_data/          # ML-ready data
├── 📁 plots/                      # Data visualizations
├── 📁 ml_results/                 # ML model results
├── 📁 saved_models/               # Trained models
├── 📄 asl_data_collector.py       # Advanced data collection
├── 📄 asl_train.py                # Simple data collection
├── 📄 data_preprocessing.py       # Data preprocessing pipeline
├── 📄 asl_ml_model.py             # ML training & prediction
├── 📄 asl_deep_learning.py        # Deep learning training
├── 📄 pc_asl_tts.py               # Real-time prediction
├── 📄 data_collection_config.py   # Configuration settings
├── 📄 asl_system_flowchart.py     # Flowchart generator
├── 📄 COMPLETE_INFORMATION_FEEDING_SUMMARY.md
├── 📄 DATA_COLLECTION_GUIDE.md
├── 📄 INFORMATION_FEEDING_GUIDE.md
├── 📄 asl_system_flowchart.png    # System workflow diagram
├── 📄 requirements.txt            # Python dependencies
└── 📄 README.md                   # This file
```

## 🎯 Supported ASL Gestures

### **Current Support: A-J**
| Letter | Flex Pattern | Description |
|--------|-------------|-------------|
| A | [0,1,1,1,1] | Thumb straight, others bent |
| B | [0,0,0,0,0] | All fingers straight |
| C | [0,0,1,1,1] | First two straight, others bent |
| D | [0,1,1,1,0] | Thumb and pinky straight |
| E | [1,1,1,1,1] | All fingers bent |
| F | [0,0,0,1,1] | First three straight |
| G | [0,0,0,0,1] | All but pinky straight |
| H | [0,0,0,1,0] | Middle and ring bent |
| I | [1,1,1,1,0] | Pinky straight |
| J | [1,1,1,1,0] | Pinky straight (with motion) |

### **Extensible Gesture System**
- Easy to add new gestures in `data_collection_config.py`
- Support for numbers 0-9
- Custom words and phrases
- Dynamic gesture recognition

## 🔍 Data Analysis & Visualization

### **Real-time Visualization**
- **Flex Sensors**: Live bar charts showing finger positions
- **IMU Data**: Accelerometer and gyroscope readings
- **Stability Indicators**: Visual feedback for data quality

### **Post-processing Analysis**
- **Feature Distributions**: Data spread for each sensor
- **Correlation Matrix**: Relationships between sensors
- **Label Distribution**: Frequency of each gesture
- **Quality Metrics**: Data validation and statistics

## 🚀 Advanced Features

### **Information Feeding System**
- **Stage-by-stage guidance**: Detailed instructions for each system stage
- **Validation checkpoints**: Quality control at every step
- **Troubleshooting support**: Common issues and solutions
- **Performance metrics**: Success indicators and benchmarks

### **Configuration Management**
- **Centralized settings**: All parameters in config files
- **Flexible customization**: Easy to modify for different setups
- **Hardware adaptation**: Support for various sensor configurations
- **Scalable architecture**: Easy to extend and modify

### **Quality Assurance**
- **Data validation**: Range checking and stability verification
- **Model evaluation**: Comprehensive performance metrics
- **Real-time monitoring**: Live system status and feedback
- **Error handling**: Robust error detection and recovery

## 📚 Comprehensive Documentation

### **Guides & Tutorials**
- **`COMPLETE_INFORMATION_FEEDING_SUMMARY.md`**: Complete system overview
- **`DATA_COLLECTION_GUIDE.md`**: Detailed data collection instructions
- **`INFORMATION_FEEDING_GUIDE.md`**: Step-by-step information feeding
- **`asl_system_flowchart.png`**: Visual system workflow

### **Configuration Files**
- **`data_collection_config.py`**: Data collection settings
- **`requirements.txt`**: Python dependencies
- **`esp32_asl/esp32_asl.ino`**: Enhanced ESP32 firmware

### **Scripts & Tools**
- **`asl_data_collector.py`**: Advanced data collection with visualization
- **`asl_train.py`**: Simple data collection
- **`data_preprocessing.py`**: Comprehensive data preprocessing
- **`asl_ml_model.py`**: ML training and evaluation
- **`asl_deep_learning.py`**: Deep learning implementation
- **`pc_asl_tts.py`**: Real-time prediction with TTS

## 🔧 Troubleshooting & Support

### **Common Issues**
1. **Connection Problems**: Check COM port and USB connection
2. **Sensor Issues**: Verify wiring and power supply
3. **Data Quality**: Recalibrate sensors and check stability
4. **Model Performance**: Collect more data or adjust parameters

### **Validation Checklist**
- [ ] Hardware connections verified
- [ ] Firmware uploaded successfully
- [ ] Sensors calibrated
- [ ] Data collected (50+ samples per gesture)
- [ ] Data preprocessed and validated
- [ ] Models trained with good accuracy (>90%)
- [ ] Real-time prediction working
- [ ] Text-to-speech output functional

## 🎯 Success Metrics

### **System Performance**
- **Data Quality**: >95% stable readings
- **Model Accuracy**: >90% classification accuracy
- **Real-time Performance**: <500ms prediction latency
- **System Reliability**: >99% uptime during operation

### **Information Feeding Quality**
- **Hardware**: All sensors functioning within expected ranges
- **Firmware**: Stable communication and calibration
- **Data Collection**: Sufficient samples per gesture (50+)
- **Preprocessing**: Clean, scaled, and validated data
- **ML Training**: High accuracy with good generalization
- **Prediction**: Reliable real-time recognition

## 🚀 Future Enhancements

1. **Expanded Gesture Set**: Add more ASL letters, numbers, and words
2. **Advanced ML Models**: Implement CNN, LSTM, and transformer models
3. **Mobile Integration**: Smartphone app for portable use
4. **Cloud Processing**: Upload data for continuous learning
5. **Multi-language Support**: Extend to other sign languages
6. **Gesture Sequences**: Support for complete sentences and phrases
7. **Accessibility Features**: Voice feedback and haptic responses

## 📝 Technical Notes

- **Sensor Calibration**: Essential for accurate readings
- **Data Quality**: More real data improves model performance
- **Model Selection**: Choose based on dataset size and complexity
- **Real-time Optimization**: Balance accuracy vs. speed
- **Hardware Variations**: Calibrate for your specific sensor setup

## 🤝 Contributing

This project is designed to be easily extensible:
- Add new gestures in the configuration
- Implement additional ML models
- Enhance the visualization system
- Improve the hardware design
- Add new data sources

## 📄 License

This project is open source and available under the MIT License.

---

**🎯 Ready to start?** Follow the Quick Start guide above, or dive deep into the comprehensive documentation for advanced usage and customization. 