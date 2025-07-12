import pandas as pd
import numpy as np
import requests
import zipfile
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class ASLDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.data_dir = "asl_datasets"
        
    def download_kaggle_asl_dataset(self):
        """
        Download ASL dataset from Kaggle (requires kaggle API)
        Alternative: Manual download from https://www.kaggle.com/datasets/grassknoted/asl-alphabet
        """
        print("Downloading ASL dataset from Kaggle...")
        try:
            # You'll need to install kaggle: pip install kaggle
            # And set up your API credentials
            os.system("kaggle datasets download -d grassknoted/asl-alphabet -p ./asl_datasets")
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please download manually from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
    
    def load_custom_sensor_data(self, csv_file='asl_training_data.csv'):
        """
        Load and preprocess your custom sensor data from ESP32
        """
        print(f"Loading custom sensor data from {csv_file}...")
        
        try:
            df = pd.read_csv(csv_file, header=None, names=['label', 'sensor_data'])
            
            # Parse sensor data (assuming format: "flex1,flex2,flex3,flex4,flex5,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z")
            sensor_features = []
            for data in df['sensor_data']:
                try:
                    values = [float(x.strip()) for x in data.split(',')]
                    sensor_features.append(values)
                except:
                    # Handle malformed data
                    sensor_features.append([0] * 11)  # 11 features: 5 flex + 6 IMU
            
            # Create feature columns
            feature_names = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5', 
                           'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            
            sensor_df = pd.DataFrame(sensor_features, columns=feature_names)
            sensor_df['label'] = df['label']
            
            print(f"Loaded {len(sensor_df)} samples with {len(feature_names)} features")
            return sensor_df
            
        except FileNotFoundError:
            print(f"File {csv_file} not found. Please run asl_train.py first to collect data.")
            return None
    
    def create_synthetic_data(self, num_samples=1000):
        """
        Create synthetic ASL data for testing and development
        """
        print("Creating synthetic ASL sensor data...")
        
        # Define ASL letter patterns (simplified)
        asl_patterns = {
            'A': {'flex': [0, 1, 1, 1, 1], 'acc_range': (8, 10)},
            'B': {'flex': [0, 0, 0, 0, 0], 'acc_range': (9, 11)},
            'C': {'flex': [0, 0, 1, 1, 1], 'acc_range': (8, 10)},
            'D': {'flex': [0, 1, 1, 1, 0], 'acc_range': (9, 11)},
            'E': {'flex': [1, 1, 1, 1, 1], 'acc_range': (8, 10)},
            'F': {'flex': [0, 0, 0, 1, 1], 'acc_range': (9, 11)},
            'G': {'flex': [0, 0, 0, 0, 1], 'acc_range': (8, 10)},
            'H': {'flex': [0, 0, 0, 1, 0], 'acc_range': (9, 11)},
            'I': {'flex': [1, 1, 1, 1, 0], 'acc_range': (8, 10)},
            'J': {'flex': [1, 1, 1, 1, 0], 'acc_range': (9, 11)},
        }
        
        data = []
        samples_per_letter = num_samples // len(asl_patterns)
        
        for letter, pattern in asl_patterns.items():
            for _ in range(samples_per_letter):
                # Generate flex sensor values with noise
                flex_values = []
                for flex_state in pattern['flex']:
                    if flex_state == 0:  # Straight finger
                        flex_values.append(np.random.normal(1500, 200))  # Lower values
                    else:  # Bent finger
                        flex_values.append(np.random.normal(2500, 200))  # Higher values
                
                # Generate accelerometer values
                acc_x = np.random.normal(0, 0.5)
                acc_y = np.random.normal(0, 0.5)
                acc_z = np.random.normal(pattern['acc_range'][0], 0.5)
                
                # Generate gyroscope values
                gyro_x = np.random.normal(0, 0.1)
                gyro_y = np.random.normal(0, 0.1)
                gyro_z = np.random.normal(0, 0.1)
                
                data.append([letter] + flex_values + [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        
        columns = ['label', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5', 
                  'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        df = pd.DataFrame(data, columns=columns)
        print(f"Created {len(df)} synthetic samples")
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the data for machine learning
        """
        print("Preprocessing data...")
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col != 'label']
        X = df[feature_columns]
        y = df['label']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def visualize_data(self, df):
        """
        Create visualizations of the data
        """
        print("Creating data visualizations...")
        
        # Create output directory
        os.makedirs('plots', exist_ok=True)
        
        # 1. Label distribution
        plt.figure(figsize=(12, 6))
        df['label'].value_counts().plot(kind='bar')
        plt.title('Distribution of ASL Letters')
        plt.xlabel('Letter')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/label_distribution.png')
        plt.close()
        
        # 2. Feature correlations
        feature_cols = [col for col in df.columns if col != 'label']
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[feature_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('plots/feature_correlations.png')
        plt.close()
        
        # 3. Feature distributions
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(feature_cols):
            axes[i].hist(df[col], bins=30, alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('plots/feature_distributions.png')
        plt.close()
        
        print("Visualizations saved to 'plots' directory")
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, feature_columns):
        """
        Save preprocessed data for later use
        """
        print("Saving preprocessed data...")
        
        os.makedirs('preprocessed_data', exist_ok=True)
        
        # Save as numpy arrays
        np.save('preprocessed_data/X_train.npy', X_train)
        np.save('preprocessed_data/X_test.npy', X_test)
        np.save('preprocessed_data/y_train.npy', y_train)
        np.save('preprocessed_data/y_test.npy', y_test)
        
        # Save feature names and label mapping
        feature_df = pd.DataFrame({'feature_name': feature_columns})
        feature_df.to_csv('preprocessed_data/feature_names.csv', index=False)
        
        label_mapping = pd.DataFrame({
            'encoded_label': range(len(self.label_encoder.classes_)),
            'original_label': self.label_encoder.classes_
        })
        label_mapping.to_csv('preprocessed_data/label_mapping.csv', index=False)
        
        print("Preprocessed data saved to 'preprocessed_data' directory")

    def load_all_available_data(self):
        """
        Load and combine all available sensor datasets.
        """
        print("Loading all available datasets...")
        all_dfs = []

        # 1. Custom sensor data
        custom_df = self.load_custom_sensor_data()
        if custom_df is not None:
            all_dfs.append(custom_df)
            print(f"Loaded {len(custom_df)} samples from your custom data.")

        # 2. Preprocessed sensor dataset from downloader
        preprocessed_sensor_path = 'asl_datasets/preprocessed_sensor_asl.csv'
        if os.path.exists(preprocessed_sensor_path):
            try:
                df = pd.read_csv(preprocessed_sensor_path)
                all_dfs.append(df)
                print(f"Loaded {len(df)} samples from {preprocessed_sensor_path}.")
            except Exception as e:
                print(f"Could not load {preprocessed_sensor_path}: {e}")

        # 3. Synthetic data generated by this script
        synthetic_data_path = 'synthetic_asl_data.csv'
        if os.path.exists(synthetic_data_path):
            try:
                df = pd.read_csv(synthetic_data_path)
                all_dfs.append(df)
                print(f"Loaded {len(df)} samples from {synthetic_data_path}.")
            except Exception as e:
                print(f"Could not load {synthetic_data_path}: {e}")
        
        if not all_dfs:
            print("No datasets found. Creating new synthetic data for demonstration.")
            df = self.create_synthetic_data(num_samples=2000)
            df.to_csv('synthetic_asl_data.csv', index=False)
            return df

        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Drop duplicates to ensure unique samples
        combined_df.drop_duplicates(inplace=True)
        
        print(f"Combined a total of {len(combined_df)} unique samples from {len(all_dfs)} source(s).")
        return combined_df

def main():
    """
    Main function to run the data preprocessing pipeline
    """
    preprocessor = ASLDataPreprocessor()
    
    # Load and combine all available datasets
    df = preprocessor.load_all_available_data()
    
    if df is not None and not df.empty:
        # Preprocess the data
        X_train, X_test, y_train, y_test, feature_columns = preprocessor.preprocess_data(df)
        
        # Create visualizations
        preprocessor.visualize_data(df)
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data(X_train, X_test, y_train, y_test, feature_columns)
        
        print("\nData preprocessing completed!")
        print("Next steps:")
        print("1. Check the 'plots' directory for data visualizations.")
        print("2. Use the preprocessed data in 'preprocessed_data' directory for training your model.")
    else:
        print("\nNo data was available to process.")
        print("Please run `python asl_train.py` to collect your own data, or")
        print("run `python dataset_downloader.py` to generate sample datasets.")

if __name__ == "__main__":
    main() 