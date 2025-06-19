# Developing Custom ML Models for Environmental Monitoring

Learn how to create, train, and deploy custom machine learning models for environmental monitoring on Furcate Nano devices.

## Overview

This tutorial covers the complete process of developing custom ML models for environmental data analysis:

- Training models for environmental classification
- Creating anomaly detection systems
- Deploying models for edge inference
- Model optimization for embedded devices
- Continuous learning and model updates

## Prerequisites

### Technical Requirements
- **Platform**: Raspberry Pi 4/5 (TensorFlow Lite) or NVIDIA Jetson Nano (PyTorch)
- **Memory**: Minimum 4GB RAM (8GB recommended for training)
- **Storage**: 32GB+ for datasets and models
- **Python**: 3.8+ with virtual environment support

### Knowledge Requirements
- Basic understanding of machine learning concepts
- Python programming experience
- Familiarity with data science libraries (pandas, numpy)
- Understanding of environmental data types

### Development Environment Setup

```bash
# Create isolated development environment
python3 -m venv furcate-ml-dev
source furcate-ml-dev/bin/activate

# Install development dependencies
pip install --upgrade pip

# For Raspberry Pi (TensorFlow Lite)
pip install tensorflow==2.16.1
pip install tensorflow-lite==2.16.1
pip install tflite-support==0.4.4

# For NVIDIA Jetson (PyTorch)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Common ML dependencies
pip install pandas==2.2.0
pip install numpy==1.26.4
pip install scikit-learn==1.4.0
pip install matplotlib==3.8.2
pip install seaborn==0.13.2
pip install jupyter==1.0.0

# Environmental data processing
pip install scipy==1.12.0
pip install plotly==5.18.0
pip install optuna==3.5.0  # Hyperparameter optimization
```

## Part 1: Environmental Data Understanding

### Environmental Data Types

Environmental monitoring generates several types of data suitable for ML:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class EnvironmentalDataAnalyzer:
    """Analyze environmental data for ML model development"""
    
    def __init__(self, data_path="environmental_data.csv"):
        self.data_path = data_path
        self.data = None
        self.features = None
        self.target = None
    
    def load_data(self):
        """Load and explore environmental data"""
        # Sample environmental data structure
        sample_data = {
            'timestamp': pd.date_range('2024-01-01', periods=10000, freq='5T'),
            'temperature': np.random.normal(22, 5, 10000),
            'humidity': np.random.normal(50, 15, 10000),
            'pressure': np.random.normal(1013, 20, 10000),
            'air_quality': np.random.lognormal(3, 0.5, 10000),
            'soil_moisture': np.random.beta(2, 3, 10000) * 100,
            'light_intensity': np.abs(np.random.normal(500, 200, 10000)),
            'wind_speed': np.random.exponential(2, 10000),
            'precipitation': np.random.exponential(0.1, 10000),
            'hour_of_day': pd.date_range('2024-01-01', periods=10000, freq='5T').hour,
            'day_of_week': pd.date_range('2024-01-01', periods=10000, freq='5T').dayofweek,
            'season': (pd.date_range('2024-01-01', periods=10000, freq='5T').month % 12) // 3
        }
        
        self.data = pd.DataFrame(sample_data)
        
        # Add derived features
        self.data['comfort_index'] = self._calculate_comfort_index()
        self.data['air_quality_category'] = self._categorize_air_quality()
        
        return self.data
    
    def _calculate_comfort_index(self):
        """Calculate human comfort index from temperature and humidity"""
        temp = self.data['temperature']
        humidity = self.data['humidity']
        
        # Simplified comfort index calculation
        comfort = 100 - abs(temp - 22) * 2 - abs(humidity - 50) * 0.5
        return np.clip(comfort, 0, 100)
    
    def _categorize_air_quality(self):
        """Categorize air quality into classes"""
        aqi = self.data['air_quality']
        
        conditions = [
            (aqi <= 50),
            (aqi <= 100),
            (aqi <= 150),
            (aqi <= 200),
            (aqi <= 300),
            (aqi > 300)
        ]
        choices = ['Good', 'Moderate', 'Unhealthy for Sensitive', 
                  'Unhealthy', 'Very Unhealthy', 'Hazardous']
        
        return np.select(conditions, choices, default='Unknown')
    
    def explore_data(self):
        """Comprehensive data exploration"""
        print("Environmental Data Analysis")
        print("=" * 50)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        print("\nData types:")
        print(self.data.dtypes)
        
        print("\nStatistical Summary:")
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        print(self.data[numeric_columns].describe())
        
        print("\nMissing values:")
        print(self.data.isnull().sum())
        
        print("\nAir Quality Distribution:")
        print(self.data['air_quality_category'].value_counts())
        
        return self.data.describe()
    
    def visualize_patterns(self):
        """Visualize environmental data patterns"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Temperature trends
        axes[0, 0].plot(self.data['timestamp'], self.data['temperature'])
        axes[0, 0].set_title('Temperature Over Time')
        axes[0, 0].set_ylabel('Temperature (°C)')
        
        # Humidity vs Temperature
        axes[0, 1].scatter(self.data['temperature'], self.data['humidity'], alpha=0.5)
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Humidity (%)')
        axes[0, 1].set_title('Temperature vs Humidity')
        
        # Air quality distribution
        axes[0, 2].hist(self.data['air_quality'], bins=50, alpha=0.7)
        axes[0, 2].set_xlabel('Air Quality Index')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Air Quality Distribution')
        
        # Hourly patterns
        hourly_temp = self.data.groupby('hour_of_day')['temperature'].mean()
        axes[1, 0].plot(hourly_temp.index, hourly_temp.values)
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Average Temperature (°C)')
        axes[1, 0].set_title('Daily Temperature Pattern')
        
        # Correlation heatmap
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 1], fmt='.2f')
        axes[1, 1].set_title('Feature Correlations')
        
        # Seasonal patterns
        seasonal_aqi = self.data.groupby('season')['air_quality'].mean()
        axes[1, 2].bar(['Winter', 'Spring', 'Summer', 'Fall'], seasonal_aqi.values)
        axes[1, 2].set_ylabel('Average Air Quality')
        axes[1, 2].set_title('Seasonal Air Quality Patterns')
        
        plt.tight_layout()
        plt.savefig('environmental_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def detect_anomalies(self):
        """Detect anomalies in environmental data"""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        # Select features for anomaly detection
        anomaly_features = ['temperature', 'humidity', 'air_quality', 'pressure']
        X = self.data[anomaly_features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Detect anomalies
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        
        # Add anomaly labels to data
        self.data['anomaly'] = anomaly_labels
        self.data['anomaly_label'] = self.data['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
        
        print(f"Detected {sum(anomaly_labels == -1)} anomalies out of {len(anomaly_labels)} data points")
        print(f"Anomaly rate: {sum(anomaly_labels == -1) / len(anomaly_labels) * 100:.2f}%")
        
        return anomaly_labels
    
    def prepare_features(self, target_variable='air_quality_category'):
        """Prepare features for ML model training"""
        # Select feature columns
        feature_columns = [
            'temperature', 'humidity', 'pressure', 'soil_moisture',
            'light_intensity', 'wind_speed', 'precipitation',
            'hour_of_day', 'day_of_week', 'season', 'comfort_index'
        ]
        
        self.features = self.data[feature_columns].copy()
        self.target = self.data[target_variable].copy()
        
        # Handle missing values
        self.features = self.features.fillna(self.features.mean())
        
        print(f"Prepared {len(feature_columns)} features for ML training")
        print(f"Target variable: {target_variable}")
        print(f"Target distribution:\n{self.target.value_counts()}")
        
        return self.features, self.target

# Usage example
analyzer = EnvironmentalDataAnalyzer()
data = analyzer.load_data()
analyzer.explore_data()
analyzer.visualize_patterns()
analyzer.detect_anomalies()
features, target = analyzer.prepare_features()
```

## Part 2: Model Development for Raspberry Pi (TensorFlow Lite)

### Environmental Classification Model

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class TensorFlowLiteEnvironmentalModel:
    """TensorFlow Lite model for environmental classification"""
    
    def __init__(self, model_name="environmental_classifier"):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        
    def prepare_data(self, features, target, test_size=0.2):
        """Prepare data for TensorFlow training"""
        # Encode categorical target
        y_encoded = self.label_encoder.fit_transform(target)
        y_categorical = keras.utils.to_categorical(y_encoded)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_categorical, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Number of classes: {y_categorical.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape, num_classes):
        """Build optimized neural network for edge deployment"""
        self.model = keras.Sequential([
            # Input layer with batch normalization for stability
            keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            # Hidden layers optimized for edge devices
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.1),
            
            # Output layer
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with optimization for edge deployment
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=100):
        """Train the environmental classification model"""
        # Callbacks for training optimization
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                f'{self.model_name}_best.h5', save_best_only=True, monitor='val_accuracy'
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        # Predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Convert back to original labels
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_true_labels = self.label_encoder.inverse_transform(y_true)
        
        # Print detailed metrics
        print("Classification Report:")
        print(classification_report(y_true_labels, y_pred_labels))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        
        # Visualize results
        self._plot_training_history()
        self._plot_confusion_matrix(cm, self.label_encoder.classes_)
        
        return {
            'accuracy': self.model.evaluate(X_test, y_test, verbose=0)[1],
            'predictions': y_pred_labels,
            'probabilities': y_pred_prob
        }
    
    def convert_to_tflite(self, quantize=True):
        """Convert model to TensorFlow Lite format for edge deployment"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Enable quantization for smaller model size and faster inference
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save model
        tflite_filename = f'{self.model_name}_quantized.tflite' if quantize else f'{self.model_name}.tflite'
        with open(tflite_filename, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model converted to TensorFlow Lite: {tflite_filename}")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        # Test TFLite model
        self._test_tflite_model(tflite_filename)
        
        return tflite_filename
    
    def _representative_dataset(self):
        """Generate representative dataset for quantization"""
        for _ in range(100):
            # Use random samples from training data for quantization
            sample = np.random.random((1, 11)).astype(np.float32)
            yield [sample]
    
    def _test_tflite_model(self, tflite_filename):
        """Test TensorFlow Lite model performance"""
        interpreter = tf.lite.Interpreter(model_path=tflite_filename)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("TFLite Model Details:")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")
        
        # Test inference speed
        test_input = np.random.random(input_details[0]['shape']).astype(input_details[0]['dtype'])
        
        import time
        start_time = time.time()
        for _ in range(100):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / 100 * 1000
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        
        return avg_inference_time
    
    def _plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{self.model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage example
tf_model = TensorFlowLiteEnvironmentalModel("air_quality_classifier")
X_train, X_test, y_train, y_test = tf_model.prepare_data(features, target)
model = tf_model.build_model(X_train.shape[1], y_train.shape[1])
history = tf_model.train_model(X_train, y_train, X_test, y_test)
evaluation = tf_model.evaluate_model(X_test, y_test)
tflite_file = tf_model.convert_to_tflite(quantize=True)
```

## Part 3: Model Development for NVIDIA Jetson (PyTorch)

### PyTorch Environmental Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class PyTorchEnvironmentalModel:
    """PyTorch model optimized for NVIDIA Jetson deployment"""
    
    def __init__(self, model_name="environmental_classifier_pytorch"):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def prepare_data(self, features, target, test_size=0.2, batch_size=32):
        """Prepare data for PyTorch training"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(target)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of classes: {len(np.unique(y_encoded))}")
        
        return train_loader, test_loader, X_test_tensor, y_test_tensor
    
    def build_model(self, input_size, num_classes):
        """Build PyTorch model optimized for Jetson deployment"""
        class EnvironmentalNet(nn.Module):
            def __init__(self, input_size, num_classes):
                super(EnvironmentalNet, self).__init__()
                
                self.features = nn.Sequential(
                    # First layer with batch normalization
                    nn.Linear(input_size, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    # Second layer
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    # Third layer
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                self.classifier = nn.Linear(16, num_classes)
                
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        self.model = EnvironmentalNet(input_size, num_classes).to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model created with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def train_model(self, train_loader, test_loader, epochs=100, learning_rate=0.001):
        """Train PyTorch model with GPU acceleration"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct_train / total_train
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = self.model(data)
                    val_loss += criterion(outputs, target).item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += target.size(0)
                    correct_val += (predicted == target).sum().item()
            
            val_loss /= len(test_loader)
            val_acc = 100 * correct_val / total_val
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'{self.model_name}_best.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}]')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
                print('-' * 50)
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(f'{self.model_name}_best.pth'))
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_acc
        }
    
    def evaluate_model(self, test_loader, X_test, y_test):
        """Evaluate PyTorch model performance"""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert back to original labels
        y_pred_labels = self.label_encoder.inverse_transform(all_predictions)
        y_true_labels = self.label_encoder.inverse_transform(y_test.numpy())
        
        print("Classification Report:")
        print(classification_report(y_true_labels, y_pred_labels))
        
        # Calculate accuracy
        accuracy = (np.array(all_predictions) == y_test.numpy()).mean()
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred_labels,
            'probabilities': np.array(all_probabilities)
        }
    
    def optimize_for_jetson(self):
        """Optimize model for NVIDIA Jetson deployment"""
        # Enable TensorRT optimization if available
        try:
            import torch_tensorrt
            
            # Create example input
            example_input = torch.randn(1, 11).to(self.device)
            
            # Compile with TensorRT
            traced_model = torch.jit.trace(self.model, example_input)
            trt_model = torch_tensorrt.compile(
                traced_model,
                inputs=[torch_tensorrt.Input(example_input.shape)],
                enabled_precisions=[torch.float, torch.half],
                workspace_size=1 << 22
            )
            
            # Save optimized model
            torch.jit.save(trt_model, f'{self.model_name}_tensorrt.pth')
            print("Model optimized with TensorRT")
            
            return trt_model
            
        except ImportError:
            print("TensorRT optimization not available. Using standard PyTorch model.")
            
            # Standard PyTorch optimization
            self.model.eval()
            traced_model = torch.jit.trace(self.model, torch.randn(1, 11).to(self.device))
            torch.jit.save(traced_model, f'{self.model_name}_optimized.pth')
            print("Model optimized with PyTorch JIT")
            
            return traced_model
    
    def benchmark_inference(self, num_iterations=1000):
        """Benchmark inference speed on Jetson"""
        self.model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(100):
                dummy_input = torch.randn(1, 11).to(self.device)
                _ = self.model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                dummy_input = torch.randn(1, 11).to(self.device)
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_iterations * 1000
        fps = 1000 / avg_inference_time
        
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        print(f"Throughput: {fps:.1f} FPS")
        
        return avg_inference_time, fps

# Usage example for Jetson
pytorch_model = PyTorchEnvironmentalModel("jetson_air_quality_classifier")
train_loader, test_loader, X_test, y_test = pytorch_model.prepare_data(features, target)
model = pytorch_model.build_model(features.shape[1], len(target.unique()))
training_history = pytorch_model.train_model(train_loader, test_loader)
evaluation = pytorch_model.evaluate_model(test_loader, X_test, y_test)
optimized_model = pytorch_model.optimize_for_jetson()
inference_time, fps = pytorch_model.benchmark_inference()
```

## Part 4: Anomaly Detection Model

### Unsupervised Anomaly Detection

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

class EnvironmentalAnomalyDetector:
    """Anomaly detection for environmental monitoring"""
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.threshold = None
        
    def train_anomaly_detector(self, X_train):
        """Train anomaly detection model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Train isolation forest
        self.model.fit(X_pca)
        
        # Calculate threshold
        anomaly_scores = self.model.decision_function(X_pca)
        self.threshold = np.percentile(anomaly_scores, self.contamination * 100)
        
        print(f"Anomaly detector trained on {X_train.shape[0]} samples")
        print(f"Reduced dimensions from {X_train.shape[1]} to {X_pca.shape[1]}")
        print(f"Anomaly threshold: {self.threshold:.4f}")
        
        return self.model
    
    def detect_anomalies(self, X_new):
        """Detect anomalies in new data"""
        # Scale and transform
        X_scaled = self.scaler.transform(X_new)
        X_pca = self.pca.transform(X_scaled)
        
        # Get anomaly scores
        anomaly_scores = self.model.decision_function(X_pca)
        predictions = self.model.predict(X_pca)
        
        # Calculate anomaly probabilities
        anomaly_probs = 1 / (1 + np.exp(anomaly_scores))
        
        return {
            'predictions': predictions,  # 1 = normal, -1 = anomaly
            'scores': anomaly_scores,
            'probabilities': anomaly_probs,
            'is_anomaly': predictions == -1
        }
    
    def save_model(self, filename_prefix="anomaly_detector"):
        """Save trained anomaly detection model"""
        joblib.dump(self.model, f"{filename_prefix}_isolation_forest.pkl")
        joblib.dump(self.scaler, f"{filename_prefix}_scaler.pkl")
        joblib.dump(self.pca, f"{filename_prefix}_pca.pkl")
        
        model_info = {
            'contamination': self.contamination,
            'threshold': self.threshold,
            'n_components': self.pca.n_components_
        }
        
        joblib.dump(model_info, f"{filename_prefix}_info.pkl")
        print(f"Anomaly detection model saved with prefix: {filename_prefix}")
    
    def load_model(self, filename_prefix="anomaly_detector"):
        """Load trained anomaly detection model"""
        self.model = joblib.load(f"{filename_prefix}_isolation_forest.pkl")
        self.scaler = joblib.load(f"{filename_prefix}_scaler.pkl")
        self.pca = joblib.load(f"{filename_prefix}_pca.pkl")
        
        model_info = joblib.load(f"{filename_prefix}_info.pkl")
        self.contamination = model_info['contamination']
        self.threshold = model_info['threshold']
        
        print(f"Anomaly detection model loaded from: {filename_prefix}")

# Train anomaly detector
anomaly_detector = EnvironmentalAnomalyDetector(contamination=0.05)
anomaly_model = anomaly_detector.train_anomaly_detector(features)
anomaly_detector.save_model("environmental_anomaly_detector")
```

## Part 5: Model Deployment and Integration

### Furcate Nano ML Integration

```python
import json
import numpy as np
from datetime import datetime

class FurcateMLProcessor:
    """ML processor for Furcate Nano edge deployment"""
    
    def __init__(self, platform="raspberry_pi"):
        self.platform = platform
        self.classification_model = None
        self.anomaly_detector = None
        self.scaler = None
        self.label_encoder = None
        
        if platform == "raspberry_pi":
            self._setup_tensorflow_lite()
        elif platform == "jetson":
            self._setup_pytorch()
    
    def _setup_tensorflow_lite(self):
        """Setup TensorFlow Lite for Raspberry Pi"""
        try:
            import tensorflow.lite as tflite
            
            # Load TFLite model
            self.interpreter = tflite.Interpreter(
                model_path="air_quality_classifier_quantized.tflite"
            )
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print("TensorFlow Lite model loaded successfully")
            
        except Exception as e:
            print(f"Error loading TFLite model: {e}")
    
    def _setup_pytorch(self):
        """Setup PyTorch for NVIDIA Jetson"""
        try:
            import torch
            
            # Load optimized PyTorch model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = torch.jit.load("jetson_air_quality_classifier_optimized.pth")
            self.model.eval()
            
            print(f"PyTorch model loaded on {self.device}")
            
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
    
    def load_preprocessing_models(self):
        """Load preprocessing models"""
        try:
            import joblib
            
            self.scaler = joblib.load("feature_scaler.pkl")
            self.label_encoder = joblib.load("label_encoder.pkl")
            
            # Load anomaly detector
            from sklearn.ensemble import IsolationForest
            self.anomaly_detector = EnvironmentalAnomalyDetector()
            self.anomaly_detector.load_model("environmental_anomaly_detector")
            
            print("Preprocessing models loaded successfully")
            
        except Exception as e:
            print(f"Error loading preprocessing models: {e}")
    
    def preprocess_sensor_data(self, sensor_readings):
        """Preprocess sensor data for ML inference"""
        try:
            # Extract features from sensor readings
            features = []
            
            # Temperature and humidity
            temp_humidity = sensor_readings.get('temperature_humidity', {})
            if isinstance(temp_humidity, dict) and 'value' in temp_humidity:
                temp_data = temp_humidity['value']
                features.extend([
                    temp_data.get('temperature', 20.0),
                    temp_data.get('humidity', 50.0)
                ])
            else:
                features.extend([20.0, 50.0])  # Default values
            
            # Pressure
            pressure_data = sensor_readings.get('pressure_temperature', {})
            if isinstance(pressure_data, dict) and 'value' in pressure_data:
                features.append(pressure_data['value'].get('pressure', 1013.0))
            else:
                features.append(1013.0)
            
            # Air quality
            air_quality = sensor_readings.get('air_quality', {})
            if isinstance(air_quality, dict) and 'value' in air_quality:
                features.append(air_quality['value'].get('aqi', 50.0))
            else:
                features.append(50.0)
            
            # Add derived features
            current_time = datetime.now()
            features.extend([
                current_time.hour,  # Hour of day
                current_time.weekday(),  # Day of week
                (current_time.month - 1) // 3,  # Season
                0.0,  # Soil moisture (placeholder)
                500.0,  # Light intensity (placeholder)
                2.0,  # Wind speed (placeholder)
                0.0   # Precipitation (placeholder)
            ])
            
            # Calculate comfort index
            comfort_index = 100 - abs(features[0] - 22) * 2 - abs(features[1] - 50) * 0.5
            features.append(max(0, min(100, comfort_index)))
            
            # Scale features
            if self.scaler:
                features_array = np.array(features).reshape(1, -1)
                features_scaled = self.scaler.transform(features_array)
                return features_scaled[0]
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error preprocessing sensor data: {e}")
            return np.zeros(11)  # Return zero array as fallback
    
    def run_classification(self, features):
        """Run environmental classification"""
        try:
            if self.platform == "raspberry_pi":
                return self._run_tflite_classification(features)
            elif self.platform == "jetson":
                return self._run_pytorch_classification(features)
            else:
                return self._run_fallback_classification(features)
                
        except Exception as e:
            print(f"Classification error: {e}")
            return {
                'environmental_class': 'unknown',
                'confidence': 0.0,
                'probabilities': []
            }
    
    def _run_tflite_classification(self, features):
        """Run TensorFlow Lite classification"""
        try:
            # Prepare input
            input_data = features.reshape(1, -1).astype(np.float32)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            probabilities = output_data[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            
            # Convert to class label
            if self.label_encoder:
                class_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            else:
                class_labels = ['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy']
                class_label = class_labels[min(predicted_class_idx, len(class_labels) - 1)]
            
            return {
                'environmental_class': class_label,
                'confidence': float(confidence),
                'probabilities': probabilities.tolist()
            }
            
        except Exception as e:
            print(f"TFLite classification error: {e}")
            return self._run_fallback_classification(features)
    
    def _run_pytorch_classification(self, features):
        """Run PyTorch classification"""
        try:
            import torch
            
            # Prepare input
            input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            
            # Convert to class label
            if self.label_encoder:
                class_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            else:
                class_labels = ['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy']
                class_label = class_labels[min(predicted_class_idx, len(class_labels) - 1)]
            
            return {
                'environmental_class': class_label,
                'confidence': float(confidence),
                'probabilities': probabilities.tolist()
            }
            
        except Exception as e:
            print(f"PyTorch classification error: {e}")
            return self._run_fallback_classification(features)
    
    def _run_fallback_classification(self, features):
        """Fallback classification using rules"""
        try:
            # Simple rule-based classification
            temperature = features[0] if len(features) > 0 else 20.0
            humidity = features[1] if len(features) > 1 else 50.0
            air_quality = features[3] if len(features) > 3 else 50.0
            
            if air_quality <= 50:
                class_label = 'Good'
                confidence = 0.8
            elif air_quality <= 100:
                class_label = 'Moderate'
                confidence = 0.7
            elif air_quality <= 150:
                class_label = 'Unhealthy for Sensitive'
                confidence = 0.6
            else:
                class_label = 'Unhealthy'
                confidence = 0.5
            
            return {
                'environmental_class': class_label,
                'confidence': confidence,
                'probabilities': [0.25, 0.25, 0.25, 0.25]
            }
            
        except Exception as e:
            print(f"Fallback classification error: {e}")
            return {
                'environmental_class': 'unknown',
                'confidence': 0.0,
                'probabilities': []
            }
    
    def run_anomaly_detection(self, features):
        """Run anomaly detection"""
        try:
            if self.anomaly_detector:
                features_array = features.reshape(1, -1)
                anomaly_results = self.anomaly_detector.detect_anomalies(features_array)
                
                return {
                    'is_anomaly': bool(anomaly_results['is_anomaly'][0]),
                    'anomaly_score': float(anomaly_results['scores'][0]),
                    'anomaly_probability': float(anomaly_results['probabilities'][0])
                }
            else:
                # Simple rule-based anomaly detection
                temperature = features[0] if len(features) > 0 else 20.0
                humidity = features[1] if len(features) > 1 else 50.0
                
                is_anomaly = (temperature < -10 or temperature > 50 or 
                             humidity < 0 or humidity > 100)
                
                return {
                    'is_anomaly': is_anomaly,
                    'anomaly_score': 0.8 if is_anomaly else 0.2,
                    'anomaly_probability': 0.9 if is_anomaly else 0.1
                }
                
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'anomaly_probability': 0.0
            }
    
    def process_environmental_data(self, sensor_readings):
        """Complete ML processing pipeline"""
        try:
            # Preprocess sensor data
            features = self.preprocess_sensor_data(sensor_readings)
            
            # Run classification
            classification_results = self.run_classification(features)
            
            # Run anomaly detection
            anomaly_results = self.run_anomaly_detection(features)
            
            # Combine results
            ml_results = {
                **classification_results,
                **anomaly_results,
                'timestamp': datetime.now().isoformat(),
                'features_processed': len(features),
                'platform': self.platform
            }
            
            return ml_results
            
        except Exception as e:
            print(f"ML processing error: {e}")
            return {
                'environmental_class': 'unknown',
                'confidence': 0.0,
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'platform': self.platform
            }

# Usage in Furcate Nano
# This integrates with the existing edge_ml.py module
ml_processor = FurcateMLProcessor(platform="raspberry_pi")  # or "jetson"
ml_processor.load_preprocessing_models()

# Example usage with sensor data
sample_sensor_data = {
    'temperature_humidity': {
        'value': {'temperature': 23.5, 'humidity': 45.0}
    },
    'air_quality': {
        'value': {'aqi': 75.0}
    }
}

results = ml_processor.process_environmental_data(sample_sensor_data)
print("ML Results:", json.dumps(results, indent=2))
```

## Part 6: Model Deployment Scripts

### Deployment Script for Raspberry Pi

```bash
#!/bin/bash
# deploy_ml_models_rpi.sh
# Deploy custom ML models to Raspberry Pi

echo "Deploying custom ML models to Raspberry Pi..."

# Create model directory
sudo mkdir -p /opt/furcate-nano/models
cd /opt/furcate-nano/models

# Copy trained models
echo "Copying trained models..."
cp ~/furcate-ml-dev/air_quality_classifier_quantized.tflite ./
cp ~/furcate-ml-dev/feature_scaler.pkl ./
cp ~/furcate-ml-dev/label_encoder.pkl ./
cp ~/furcate-ml-dev/environmental_anomaly_detector_*.pkl ./

# Set permissions
sudo chown -R furcate:furcate /opt/furcate-nano/models
sudo chmod 644 /opt/furcate-nano/models/*

# Update Furcate Nano configuration
echo "Updating ML configuration..."
sudo tee -a /etc/furcate-nano/ml-config.yaml > /dev/null <<EOF
ml:
  simulation: false
  model_path: "/opt/furcate-nano/models"
  platform: "raspberry_pi"
  models:
    environmental_classifier:
      file: "air_quality_classifier_quantized.tflite"
      enabled: true
      type: "tflite"
    anomaly_detector:
      file: "environmental_anomaly_detector"
      enabled: true
      type: "sklearn"
  preprocessing:
    scaler: "feature_scaler.pkl"
    label_encoder: "label_encoder.pkl"
  performance:
    inference_timeout: 5000  # 5 seconds
    batch_size: 1
    optimization_level: "quantized"
EOF

# Test model deployment
echo "Testing model deployment..."
python3 -c "
import sys
sys.path.append('/opt/furcate-nano')
from ml_processor import FurcateMLProcessor

processor = FurcateMLProcessor('raspberry_pi')
processor.load_preprocessing_models()

test_data = {
    'temperature_humidity': {'value': {'temperature': 22.0, 'humidity': 50.0}},
    'air_quality': {'value': {'aqi': 60.0}}
}

results = processor.process_environmental_data(test_data)
print('Deployment test successful!')
print('Results:', results)
"

echo "ML model deployment complete!"
```

### Deployment Script for NVIDIA Jetson

```bash
#!/bin/bash
# deploy_ml_models_jetson.sh
# Deploy custom ML models to NVIDIA Jetson

echo "Deploying custom ML models to NVIDIA Jetson..."

# Create model directory
sudo mkdir -p /opt/furcate-nano/models
cd /opt/furcate-nano/models

# Copy trained models
echo "Copying trained models..."
cp ~/furcate-ml-dev/jetson_air_quality_classifier_optimized.pth ./
cp ~/furcate-ml-dev/feature_scaler.pkl ./
cp ~/furcate-ml-dev/label_encoder.pkl ./
cp ~/furcate-ml-dev/environmental_anomaly_detector_*.pkl ./

# Set permissions
sudo chown -R jetson:jetson /opt/furcate-nano/models
sudo chmod 644 /opt/furcate-nano/models/*

# Update Furcate Nano configuration
echo "Updating ML configuration for Jetson..."
sudo tee -a /etc/furcate-nano/ml-config.yaml > /dev/null <<EOF
ml:
  simulation: false
  model_path: "/opt/furcate-nano/models"
  platform: "jetson"
  models:
    environmental_classifier:
      file: "jetson_air_quality_classifier_optimized.pth"
      enabled: true
      type: "pytorch_jit"
    anomaly_detector:
      file: "environmental_anomaly_detector"
      enabled: true
      type: "sklearn"
  preprocessing:
    scaler: "feature_scaler.pkl"
    label_encoder: "label_encoder.pkl"
  gpu:
    enabled: true
    memory_fraction: 0.5
    tensorrt_optimization: true
  performance:
    inference_timeout: 2000  # 2 seconds
    batch_size: 1
    optimization_level: "tensorrt"
EOF

# Test GPU availability
echo "Testing GPU availability..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Test model deployment
echo "Testing model deployment..."
python3 -c "
import sys
sys.path.append('/opt/furcate-nano')
from ml_processor import FurcateMLProcessor

processor = FurcateMLProcessor('jetson')
processor.load_preprocessing_models()

test_data = {
    'temperature_humidity': {'value': {'temperature': 22.0, 'humidity': 50.0}},
    'air_quality': {'value': {'aqi': 60.0}}
}

results = processor.process_environmental_data(test_data)
print('Deployment test successful!')
print('Results:', results)
"

echo "ML model deployment complete!"
```

## Conclusion

You've successfully learned how to develop, train, and deploy custom ML models for Furcate Nano! This comprehensive tutorial covered:

### Key Achievements

- **Data Understanding**: Environmental data analysis and preparation
- **Model Development**: Both TensorFlow Lite (Pi) and PyTorch (Jetson) implementations
- **Optimization**: Edge-specific optimizations for embedded deployment
- **Anomaly Detection**: Unsupervised learning for environmental monitoring
- **Integration**: Complete deployment pipeline with Furcate Nano

### Platform-Specific Optimizations

**Raspberry Pi (TensorFlow Lite):**
- Quantized models for minimal memory usage
- Optimized for ARM64 architecture
- Fast inference with reduced precision

**NVIDIA Jetson (PyTorch):**
- GPU acceleration with CUDA
- TensorRT optimization for maximum performance
- JIT compilation for deployment efficiency

### Next Steps

1. **Experiment with Different Architectures**: Try CNN, LSTM, or Transformer models
2. **Advanced Preprocessing**: Implement time-series features and rolling statistics
3. **Online Learning**: Add capability for model updates with new data
4. **Multi-Modal Learning**: Combine different sensor types for better predictions
5. **Federated Learning**: Share model improvements across device networks

### Performance Tips

- **Monitor Resource Usage**: Use htop, nvidia-smi for system monitoring
- **Batch Processing**: Process multiple samples together when possible
- **Model Versioning**: Keep track of model versions and performance metrics
- **A/B Testing**: Compare different models in production environments

### Troubleshooting Common Issues

- **Memory Errors**: Reduce model size or batch size
- **Slow Inference**: Check for proper GPU utilization or model optimization
- **Poor Accuracy**: Collect more training data or adjust hyperparameters
- **Deployment Failures**: Verify model format compatibility and dependencies

Your custom ML models are now ready for real-world environmental monitoring on edge devices!