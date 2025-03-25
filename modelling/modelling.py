from model.randomforest import RandomForest
import numpy as np
from typing import Dict, Tuple, List
import pandas as pd


class HierarchicalModelController:
    def __init__(self, data, df):
        self.data = data
        self.df = df
        self.models = {
            'type2': None,  # Base model for Type2
            'type3': {},    # Dictionary of Type3 models for each Type2 class
            'type4': {}     # Dictionary of Type4 models for each (Type2, Type3) combination
        }
        
    def train(self):
        """Train hierarchical models for Type2, Type3, and Type4."""
        # Train base Type2 model
        print("Training base Type2 model...")
        self.models['type2'] = RandomForest("Type2_Base", self.data.get_embeddings(), self.df)
        self.models['type2'].train(self.data)
        
        # Get Type2 predictions for training data
        type2_train_pred = self.models['type2'].predict(self.data.X_train)
        
        # Train Type3 models for each Type2 class
        print("\nTraining Type3 models for each Type2 class...")
        unique_type2 = np.unique(self.data.y_train_type2)
        for t2 in unique_type2:
            # Filter data for this Type2 class
            mask = self.data.y_train_type2 == t2
            if np.sum(mask) < 3:  # Skip if too few samples
                continue
                
            # Create filtered data
            filtered_data = self._filter_data(mask)
            
            # Train Type3 model
            model_name = f"Type3_{t2}"
            self.models['type3'][t2] = RandomForest(model_name, filtered_data.get_embeddings(), filtered_data.df)
            self.models['type3'][t2].train(filtered_data)
            
            # Get Type3 predictions for training data
            type3_train_pred = self.models['type3'][t2].predict(filtered_data.X_train)
            
            # Train Type4 models for each (Type2, Type3) combination
            print(f"\nTraining Type4 models for Type2 class: {t2}")
            unique_type3 = np.unique(filtered_data.y_train_type3)
            for t3 in unique_type3:
                # Filter data for this (Type2, Type3) combination
                mask = filtered_data.y_train_type3 == t3
                if np.sum(mask) < 3:  # Skip if too few samples
                    continue
                    
                # Create filtered data
                filtered_data_t3 = self._filter_data(mask)
                
                # Train Type4 model
                model_name = f"Type4_{t2}_{t3}"
                self.models['type4'][(t2, t3)] = RandomForest(model_name, filtered_data_t3.get_embeddings(), filtered_data_t3.df)
                self.models['type4'][(t2, t3)].train(filtered_data_t3)
    
    def predict(self, X_test):
        """Make predictions using the hierarchical models."""
        # Predict Type2
        type2_pred = self.models['type2'].predict(X_test)
        
        # Initialize prediction arrays
        type3_pred = np.zeros_like(self.data.y_test_type3)
        type4_pred = np.zeros_like(self.data.y_test_type4)
        
        # Predict Type3 for each Type2 class
        for t2 in np.unique(type2_pred):
            if t2 not in self.models['type3']:
                continue
                
            # Filter test data for this Type2 class
            mask = type2_pred == t2
            if np.sum(mask) == 0:
                continue
                
            # Get Type3 predictions
            type3_pred[mask] = self.models['type3'][t2].predict(X_test[mask])
            
            # Predict Type4 for each (Type2, Type3) combination
            for t3 in np.unique(type3_pred[mask]):
                if (t2, t3) not in self.models['type4']:
                    continue
                    
                # Filter test data for this (Type2, Type3) combination
                mask_t3 = mask & (type3_pred == t3)
                if np.sum(mask_t3) == 0:
                    continue
                    
                # Get Type4 predictions
                type4_pred[mask_t3] = self.models['type4'][(t2, t3)].predict(X_test[mask_t3])
        
        return {
            'type2': type2_pred,
            'type3': type3_pred,
            'type4': type4_pred
        }
    
    def _filter_data(self, mask):
        """Helper function to create filtered data object."""
        filtered_df = self.df[mask].copy()
        filtered_data = Data(self.data.embeddings[mask], filtered_df)
        return filtered_data


def model_predict(data, df, name):
    results = []
    print("RandomForest")
    
    # Initialize and train hierarchical model
    controller = HierarchicalModelController(data, df)
    controller.train()
    
    # Make predictions
    predictions = controller.predict(data.X_test)
    
    # Print results for each level
    print("\nType2 Classification Report:")
    print(classification_report(data.y_test_type2, predictions['type2']))
    print("\nType3 Classification Report:")
    print(classification_report(data.y_test_type3, predictions['type3']))
    print("\nType4 Classification Report:")
    print(classification_report(data.y_test_type4, predictions['type4']))
    
    # Calculate chained classification score
    score = evaluate_chained_classification(
        y_true_type2=data.y_test_type2,
        y_true_type3=data.y_test_type3,
        y_true_type4=data.y_test_type4,
        y_pred_type2=predictions['type2'],
        y_pred_type3=predictions['type3'],
        y_pred_type4=predictions['type4']
    )
    print(f"\nChained Classification Score: {score:.2f}%")


def model_evaluate(model, data):
    model.print_results(data)


def evaluate_chained_classification(y_true_type2, y_true_type3, y_true_type4,
                                 y_pred_type2, y_pred_type3, y_pred_type4):
    """
    Evaluate chained multi-output classification with partial credit scoring.
    
    Args:
        y_true_type2: True labels for Type2
        y_true_type3: True labels for Type3
        y_true_type4: True labels for Type4
        y_pred_type2: Predicted labels for Type2
        y_pred_type3: Predicted labels for Type3
        y_pred_type4: Predicted labels for Type4
        
    Returns:
        float: Average accuracy across all instances (0-100%)
    """
    scores = []
    
    for t2_true, t3_true, t4_true, t2_pred, t3_pred, t4_pred in zip(
        y_true_type2, y_true_type3, y_true_type4,
        y_pred_type2, y_pred_type3, y_pred_type4
    ):
        # If Type2 is wrong, score is 0%
        if t2_true != t2_pred:
            scores.append(0)
            continue
            
        # If Type2 is correct, give 33%
        score = 33
        
        # If Type2 and Type3 are correct, give 66%
        if t3_true == t3_pred:
            score = 66
            
            # If all three are correct, give 100%
            if t4_true == t4_pred:
                score = 100
                
        scores.append(score)
    
    return np.mean(scores)