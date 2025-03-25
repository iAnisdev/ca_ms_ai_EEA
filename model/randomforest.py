import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random
num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 df: pd.DataFrame) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.df = df
        
        # Initialize models
        self.model_type2 = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.models_type3 = {}  # Dictionary of Type3 models for each Type2 class
        self.models_type4 = {}  # Dictionary of Type4 models for each (Type2, Type3) combination
        
        self.data_transform()

    def train(self, data) -> None:
        """Train the hierarchical models."""
        if data.X_train is None or len(data.X_train) == 0:
            print("No training data available.")
            return
            
        # Print Type2 class distribution
        print("\nType2 class distribution:")
        t2_counts = pd.Series(data.y_train_type2).value_counts()
        print(t2_counts)
            
        # Train Type2 model
        self.model_type2.fit(data.X_train, data.y_train_type2)
        
        # Dictionary to store trained classes
        self.trained_type3_classes = {}
        self.trained_type4_classes = {}
        
        # Train Type3 models for each Type2 class
        for t2 in np.unique(data.y_train_type2):
            t2_str = str(t2)
            t2_mask = data.y_train_type2 == t2
            if np.sum(t2_mask) < 3:  # Skip if too few samples
                continue
                
            # Train Type3 model for this Type2 class
            X_t2 = data.X_train[t2_mask]
            y_t3 = data.y_train_type3[t2_mask]
            
            # Print Type3 class distribution for this Type2 class
            print(f"\nType3 class distribution for Type2={t2_str}:")
            t3_counts = pd.Series(y_t3).value_counts()
            print(t3_counts)
            
            # Get Type3 classes with enough samples
            trained_t3_classes = t3_counts[t3_counts >= 3].index.tolist()
            if not trained_t3_classes:
                print(f"No Type3 classes have enough samples for Type2={t2_str}")
                continue
                
            # Train Type3 model on all data for this Type2 class
            self.models_type3[t2_str] = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
            self.models_type3[t2_str].fit(X_t2, y_t3)
            self.trained_type3_classes[t2_str] = trained_t3_classes
            
            # Train Type4 models for each Type3 class
            for t3 in trained_t3_classes:
                t3_str = str(t3)
                t3_mask = y_t3 == t3
                if np.sum(t3_mask) < 3:  # Skip if too few samples
                    continue
                    
                # Train Type4 model for this (Type2, Type3) combination
                X_t3 = X_t2[t3_mask]
                y_t4 = data.y_train_type4[t2_mask][t3_mask]
                
                # Print Type4 class distribution for this Type3 class
                print(f"\nType4 class distribution for Type2={t2_str}, Type3={t3_str}:")
                t4_counts = pd.Series(y_t4).value_counts()
                print(t4_counts)
                
                # Get Type4 classes with enough samples
                trained_t4_classes = t4_counts[t4_counts >= 3].index.tolist()
                if not trained_t4_classes:
                    print(f"No Type4 classes have enough samples for Type2={t2_str}, Type3={t3_str}")
                    continue
                    
                # Train Type4 model on filtered data
                t4_mask = np.isin(y_t4, trained_t4_classes)
                if np.sum(t4_mask) < 3:
                    continue
                    
                X_t3_filtered = X_t3[t4_mask]
                y_t4_filtered = y_t4[t4_mask]
                
                key = f"{t2_str}_{t3_str}"
                self.models_type4[key] = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
                self.models_type4[key].fit(X_t3_filtered, y_t4_filtered)
                self.trained_type4_classes[key] = trained_t4_classes

    def predict(self, X_test: pd.Series):
        """Make hierarchical predictions."""
        n_samples = len(X_test)
        
        # Predict Type2
        type2_pred = self.model_type2.predict(X_test)
        
        # Initialize Type3 and Type4 predictions
        type3_pred = np.array(['Unknown'] * n_samples)
        type4_pred = np.array(['Unknown'] * n_samples)
        
        # Make Type3 predictions for each Type2 class
        for t2 in np.unique(type2_pred):
            t2_str = str(t2)
            if t2_str not in self.models_type3:
                continue
                
            t2_mask = type2_pred == t2
            if np.sum(t2_mask) == 0:
                continue
                
            # Predict Type3 for this Type2 class
            X_t2 = X_test[t2_mask]
            t3_predictions = self.models_type3[t2_str].predict(X_t2)
            
            # Only update predictions for trained Type3 classes
            valid_t3_mask = np.isin(t3_predictions, self.trained_type3_classes[t2_str])
            type3_pred[t2_mask] = np.where(valid_t3_mask, t3_predictions, 'Unknown')
            
            # Make Type4 predictions for each valid Type3 class
            for t3 in np.unique(t3_predictions[valid_t3_mask]):
                t3_str = str(t3)
                key = f"{t2_str}_{t3_str}"
                if key not in self.models_type4:
                    continue
                    
                t3_mask = t2_mask & (type3_pred == t3)
                if np.sum(t3_mask) == 0:
                    continue
                    
                # Predict Type4 for this Type3 class
                X_t3 = X_test[t3_mask]
                t4_predictions = self.models_type4[key].predict(X_t3)
                
                # Only update predictions for trained Type4 classes
                valid_t4_mask = np.isin(t4_predictions, self.trained_type4_classes[key])
                type4_pred[t3_mask] = np.where(valid_t4_mask, t4_predictions, 'Unknown')
        
        return {
            'type2': type2_pred,
            'type3': type3_pred,
            'type4': type4_pred
        }

    def print_results(self, data):
        print("\nType2 Classification Report:")
        print(classification_report(data.y_test_type2, self.predictions_type2))
        print("\nType3 Classification Report:")
        print(classification_report(data.y_test_type3, self.predictions_type3))
        print("\nType4 Classification Report:")
        print(classification_report(data.y_test_type4, self.predictions_type4))

    def data_transform(self) -> None:
        pass

