import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import serial
import time
import pyttsx3
import os

class ASLMLModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
    def load_preprocessed_data(self):
        """
        Load the preprocessed data for training
        """
        print("Loading preprocessed data...")
        
        try:
            # Load training and test data
            X_train = np.load('preprocessed_data/X_train.npy')
            X_test = np.load('preprocessed_data/X_test.npy')
            y_train = np.load('preprocessed_data/y_train.npy')
            y_test = np.load('preprocessed_data/y_test.npy')
            
            # Load feature names and label mapping
            feature_df = pd.read_csv('preprocessed_data/feature_names.csv')
            self.feature_names = feature_df['feature_name'].tolist()
            
            label_mapping = pd.read_csv('preprocessed_data/label_mapping.csv')
            self.label_encoder = dict(zip(label_mapping['encoded_label'], label_mapping['original_label']))
            
            print(f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
            print(f"Features: {len(self.feature_names)}")
            print(f"Classes: {len(self.label_encoder)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            print("Please run 'python data_preprocessing.py' first")
            return None, None, None, None
    
    def train_models(self, X_train, y_train):
        """
        Train multiple ML models for comparison
        """
        print("Training multiple ML models...")
        
        # 1. Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        self.models['RandomForest'] = rf_model
        
        # 2. Support Vector Machine
        print("Training SVM...")
        svm_model = SVC(kernel='rbf', random_state=42, probability=True)
        svm_model.fit(X_train, y_train)
        self.models['SVM'] = svm_model
        
        # 3. Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        self.models['GradientBoosting'] = gb_model
        
        # 4. Neural Network
        print("Training Neural Network...")
        nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        nn_model.fit(X_train, y_train)
        self.models['NeuralNetwork'] = nn_model
        
        print("All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        results = {}
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_test, y_test, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred
            }
            
            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
            
            # Detailed classification report
            print(f"  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=[self.label_encoder[i] for i in range(len(self.label_encoder))]))
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 BEST MODEL: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
        
        return results
    
    def create_visualizations(self, results, y_test):
        """
        Create visualizations of model performance
        """
        print("Creating performance visualizations...")
        
        os.makedirs('ml_results', exist_ok=True)
        
        # 1. Model comparison
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('ml_results/model_comparison.png')
        plt.close()
        
        # 2. Confusion matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        y_pred_best = results[best_model_name]['predictions']
        
        cm = confusion_matrix(y_test, y_pred_best)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.label_encoder[i] for i in range(len(self.label_encoder))],
                   yticklabels=[self.label_encoder[i] for i in range(len(self.label_encoder))])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('ml_results/confusion_matrix.png')
        plt.close()
        
        print("Visualizations saved to 'ml_results' directory")
    
    def save_models(self):
        """
        Save trained models for later use
        """
        print("Saving trained models...")
        
        os.makedirs('saved_models', exist_ok=True)
        
        # Save all models
        for name, model in self.models.items():
            filename = f'saved_models/{name.lower()}_model.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} model to {filename}")
        
        # Save best model separately
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x].__class__.__name__)
        best_filename = 'saved_models/best_model.pkl'
        with open(best_filename, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Saved best model ({best_model_name}) to {best_filename}")
        
        # Save model metadata
        metadata = {
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'best_model_name': best_model_name
        }
        
        with open('saved_models/model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print("All models saved successfully!")
    
    def real_time_prediction(self, model_path='saved_models/best_model.pkl'):
        """
        Real-time ASL prediction using trained model
        """
        print("Starting real-time ASL prediction...")
        
        # Load the best model
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open('saved_models/model_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            self.feature_names = metadata['feature_names']
            self.label_encoder = metadata['label_encoder']
            
        except FileNotFoundError:
            print("No trained model found. Please train a model first.")
            return
        
        # Initialize serial connection
        try:
            ser = serial.Serial('COM3', 115200, timeout=1)
            print("Connected to ESP32")
        except:
            print("Could not connect to ESP32. Make sure it's connected to COM3")
            return
        
        # Initialize text-to-speech
        engine = pyttsx3.init()
        
        print("Make ASL gestures with your glove. Press Ctrl+C to exit.")
        print("="*50)
        
        try:
            while True:
                # Read sensor data
                line = ser.readline().decode('utf-8').strip()
                if line:
                    try:
                        # Parse sensor values
                        values = [float(x.strip()) for x in line.split(',')]
                        
                        if len(values) == 11:  # 5 flex + 6 IMU
                            # Make prediction
                            prediction = model.predict([values])[0]
                            predicted_label = self.label_encoder[prediction]
                            
                            # Get confidence (probability)
                            probabilities = model.predict_proba([values])[0]
                            confidence = max(probabilities)
                            
                            # Only output if confidence is high enough
                            if confidence > 0.7:
                                print(f"Detected: {predicted_label} (Confidence: {confidence:.2f})")
                                engine.say(predicted_label)
                                engine.runAndWait()
                            
                    except ValueError:
                        continue
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        continue
                
                time.sleep(0.1)  # Small delay to prevent overwhelming
                
        except KeyboardInterrupt:
            print("\nStopping real-time prediction...")
        finally:
            ser.close()
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning for the best model
        """
        print("Performing hyperparameter tuning...")
        
        # Grid search for Random Forest
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                              rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        
        print(f"Best Random Forest parameters: {rf_grid.best_params_}")
        print(f"Best Random Forest score: {rf_grid.best_score_:.4f}")
        
        # Update the best model
        self.models['RandomForest'] = rf_grid.best_estimator_
        self.best_model = rf_grid.best_estimator_
        
        return rf_grid.best_estimator_

def main():
    """
    Main function to train and evaluate ASL ML models
    """
    print("🤖 ASL Machine Learning System")
    print("="*50)
    
    # Initialize ML system
    ml_system = ASLMLModel()
    
    # Load data
    data = ml_system.load_preprocessed_data()
    if data[0] is None:
        print("No data available. Please run data preprocessing first.")
        return
    
    X_train, X_test, y_train, y_test = data
    
    # Train models
    ml_system.train_models(X_train, y_train)
    
    # Evaluate models
    results = ml_system.evaluate_models(X_test, y_test)
    
    # Create visualizations
    ml_system.create_visualizations(results, y_test)
    
    # Save models
    ml_system.save_models()
    
    print("\n🎉 ML Training Complete!")
    print("Next steps:")
    print("1. Check 'ml_results' directory for performance visualizations")
    print("2. Use 'python asl_ml_model.py --predict' for real-time prediction")
    print("3. Models are saved in 'saved_models' directory")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--predict":
        # Run real-time prediction
        ml_system = ASLMLModel()
        ml_system.real_time_prediction()
    else:
        # Train models
        main() 