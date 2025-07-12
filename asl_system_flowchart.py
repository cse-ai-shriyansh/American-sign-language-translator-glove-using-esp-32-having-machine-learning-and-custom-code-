import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_asl_system_flowchart():
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Define colors
    colors = {
        'hardware': '#FFE6E6',      # Light red
        'firmware': '#E6F3FF',      # Light blue
        'data_collection': '#E6FFE6', # Light green
        'preprocessing': '#FFF2E6',   # Light orange
        'ml_training': '#F0E6FF',     # Light purple
        'prediction': '#FFFFE6',      # Light yellow
        'output': '#E6FFFF'           # Light cyan
    }
    
    # Title
    ax.text(10, 15.5, 'ASL Recognition System - Complete Workflow', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Hardware Section
    ax.text(2, 14.5, 'HARDWARE SETUP', fontsize=14, fontweight='bold', ha='center')
    
    # Hardware components
    hw_components = [
        (1, 13.5, 2, 0.8, 'ESP32\nMicrocontroller', colors['hardware']),
        (4, 13.5, 2, 0.8, '5x Flex Sensors\n(Thumb to Pinky)', colors['hardware']),
        (7, 13.5, 2, 0.8, 'MPU6050\nIMU Sensor', colors['hardware']),
        (10, 13.5, 2, 0.8, 'Data Glove\nAssembly', colors['hardware']),
        (13, 13.5, 2, 0.8, 'USB Connection\n(Serial)', colors['hardware'])
    ]
    
    for x, y, w, h, text, color in hw_components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Firmware Section
    ax.text(2, 12, 'FIRMWARE (ESP32)', fontsize=14, fontweight='bold', ha='center')
    
    fw_components = [
        (0.5, 11, 3, 0.8, 'Sensor\nInitialization', colors['firmware']),
        (4, 11, 3, 0.8, 'Calibration\nRoutine', colors['firmware']),
        (7.5, 11, 3, 0.8, 'Data Acquisition\n(10Hz)', colors['firmware']),
        (11, 11, 3, 0.8, 'Stability\nChecking', colors['firmware']),
        (14.5, 11, 3, 0.8, 'Serial Data\nTransmission', colors['firmware'])
    ]
    
    for x, y, w, h, text, color in fw_components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Data Collection Section
    ax.text(2, 9.5, 'DATA COLLECTION', fontsize=14, fontweight='bold', ha='center')
    
    dc_components = [
        (0.5, 8.5, 2.5, 0.8, 'Advanced\nCollector', colors['data_collection']),
        (3.5, 8.5, 2.5, 0.8, 'Real-time\nVisualization', colors['data_collection']),
        (6.5, 8.5, 2.5, 0.8, 'Gesture\nRecording', colors['data_collection']),
        (9.5, 8.5, 2.5, 0.8, 'Data\nValidation', colors['data_collection']),
        (12.5, 8.5, 2.5, 0.8, 'CSV\nExport', colors['data_collection']),
        (15.5, 8.5, 2.5, 0.8, 'Simple\nCollector', colors['data_collection'])
    ]
    
    for x, y, w, h, text, color in dc_components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Data Sources Section
    ax.text(2, 7, 'DATA SOURCES', fontsize=14, fontweight='bold', ha='center')
    
    ds_components = [
        (1, 6, 2.5, 0.8, 'Real Sensor\nData', colors['data_collection']),
        (4, 6, 2.5, 0.8, 'Synthetic\nData', colors['data_collection']),
        (7, 6, 2.5, 0.8, 'External\nDatasets', colors['data_collection']),
        (10, 6, 2.5, 0.8, 'Preprocessed\nData', colors['data_collection']),
        (13, 6, 2.5, 0.8, 'Combined\nDataset', colors['data_collection'])
    ]
    
    for x, y, w, h, text, color in ds_components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Preprocessing Section
    ax.text(2, 4.5, 'DATA PREPROCESSING', fontsize=14, fontweight='bold', ha='center')
    
    pp_components = [
        (0.5, 3.5, 2.5, 0.8, 'Data\nLoading', colors['preprocessing']),
        (3.5, 3.5, 2.5, 0.8, 'Feature\nScaling', colors['preprocessing']),
        (6.5, 3.5, 2.5, 0.8, 'Label\nEncoding', colors['preprocessing']),
        (9.5, 3.5, 2.5, 0.8, 'Train/Test\nSplit', colors['preprocessing']),
        (12.5, 3.5, 2.5, 0.8, 'Data\nVisualization', colors['preprocessing']),
        (15.5, 3.5, 2.5, 0.8, 'Feature\nSelection', colors['preprocessing'])
    ]
    
    for x, y, w, h, text, color in pp_components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # ML Training Section
    ax.text(2, 2, 'MACHINE LEARNING', fontsize=14, fontweight='bold', ha='center')
    
    ml_components = [
        (0.5, 1, 2.5, 0.8, 'Random\nForest', colors['ml_training']),
        (3.5, 1, 2.5, 0.8, 'SVM\nClassifier', colors['ml_training']),
        (6.5, 1, 2.5, 0.8, 'Gradient\nBoosting', colors['ml_training']),
        (9.5, 1, 2.5, 0.8, 'Neural\nNetwork', colors['ml_training']),
        (12.5, 1, 2.5, 0.8, 'Deep\nLearning', colors['ml_training']),
        (15.5, 1, 2.5, 0.8, 'Model\nEvaluation', colors['ml_training'])
    ]
    
    for x, y, w, h, text, color in ml_components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Prediction Section
    ax.text(2, -0.5, 'REAL-TIME PREDICTION', fontsize=14, fontweight='bold', ha='center')
    
    pred_components = [
        (1, -1.5, 2.5, 0.8, 'Live Sensor\nData', colors['prediction']),
        (4, -1.5, 2.5, 0.8, 'Feature\nExtraction', colors['prediction']),
        (7, -1.5, 2.5, 0.8, 'Model\nPrediction', colors['prediction']),
        (10, -1.5, 2.5, 0.8, 'Gesture\nClassification', colors['prediction']),
        (13, -1.5, 2.5, 0.8, 'Text-to-Speech\nOutput', colors['prediction'])
    ]
    
    for x, y, w, h, text, color in pred_components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add arrows showing flow
    arrows = [
        # Hardware to Firmware
        (2, 13.1, 2, 11.8),
        (5, 13.1, 5.5, 11.8),
        (8, 13.1, 9, 11.8),
        (11, 13.1, 12.5, 11.8),
        (14, 13.1, 16, 11.8),
        
        # Firmware to Data Collection
        (2, 10.2, 2, 9.3),
        (5.5, 10.2, 5.5, 9.3),
        (9, 10.2, 9, 9.3),
        (12.5, 10.2, 12.5, 9.3),
        (16, 10.2, 16, 9.3),
        
        # Data Collection to Data Sources
        (2, 7.7, 2, 6.8),
        (5.5, 7.7, 5.5, 6.8),
        (9, 7.7, 9, 6.8),
        (12.5, 7.7, 12.5, 6.8),
        (16, 7.7, 16, 6.8),
        
        # Data Sources to Preprocessing
        (2, 5.2, 2, 4.3),
        (5.5, 5.2, 5.5, 4.3),
        (9, 5.2, 9, 4.3),
        (12.5, 5.2, 12.5, 4.3),
        (16, 5.2, 16, 4.3),
        
        # Preprocessing to ML
        (2, 2.7, 2, 1.8),
        (5.5, 2.7, 5.5, 1.8),
        (9, 2.7, 9, 1.8),
        (12.5, 2.7, 12.5, 1.8),
        (16, 2.7, 16, 1.8),
        
        # ML to Prediction
        (2, 0.2, 2, -0.7),
        (5.5, 0.2, 5.5, -0.7),
        (9, 0.2, 9, -0.7),
        (12.5, 0.2, 12.5, -0.7),
        (16, 0.2, 16, -0.7),
    ]
    
    for x1, y1, x2, y2 in arrows:
        arrow = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black", linewidth=2)
        ax.add_patch(arrow)
    
    # Add side notes
    ax.text(18, 13, 'Hardware\nComponents', fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    ax.text(18, 11, 'ESP32\nFirmware', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    ax.text(18, 8.5, 'Data\nCollection', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    ax.text(18, 6, 'Data\nSources', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    ax.text(18, 3.5, 'Data\nProcessing', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    ax.text(18, 1, 'ML\nTraining', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    ax.text(18, -1.5, 'Real-time\nPrediction', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['hardware'], label='Hardware'),
        patches.Patch(color=colors['firmware'], label='Firmware'),
        patches.Patch(color=colors['data_collection'], label='Data Collection'),
        patches.Patch(color=colors['preprocessing'], label='Preprocessing'),
        patches.Patch(color=colors['ml_training'], label='ML Training'),
        patches.Patch(color=colors['prediction'], label='Prediction'),
        patches.Patch(color=colors['output'], label='Output')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('asl_system_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("Flowchart saved as 'asl_system_flowchart.png'")

if __name__ == "__main__":
    create_asl_system_flowchart() 