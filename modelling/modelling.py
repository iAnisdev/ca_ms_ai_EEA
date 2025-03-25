from model.randomforest import RandomForest
import numpy as np
from typing import Dict, Tuple, List
import pandas as pd
from modelling.data_model import Data
from sklearn.metrics import classification_report


def predict_hierarchical(model_chain: Dict, X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Perform hierarchical prediction using the model chain.
    
    Args:
        model_chain: Dictionary containing the hierarchical models:
            - 'type2': Base model for Type2 prediction
            - 'type3': Dictionary of Type3 models for each Type2 class
            - 'type4': Dictionary of Type4 models for each (Type2, Type3) combination
        X: Input features to predict on
        
    Returns:
        Dictionary containing predictions for each type:
            - 'type2': Type2 predictions
            - 'type3': Type3 predictions
            - 'type4': Type4 predictions
    """
    # Step 1: Predict Type2 using base model
    base_predictions = model_chain['type2'].predict(X)
    type2_pred = base_predictions['type2']
    
    # Initialize prediction arrays
    n_samples = len(X)
    type3_pred = np.array(['Unknown'] * n_samples)
    type4_pred = np.array(['Unknown'] * n_samples)
    
    # Step 2: For each predicted Type2 class, predict Type3
    for t2 in np.unique(type2_pred):
        t2_str = str(t2)
        if t2_str not in model_chain['type3']:
            continue
            
        # Get samples predicted as this Type2 class
        t2_mask = type2_pred == t2
        if np.sum(t2_mask) == 0:
            continue
            
        # Predict Type3 for these samples
        t3_predictions = model_chain['type3'][t2_str].predict(X[t2_mask])
        type3_pred[t2_mask] = t3_predictions['type3']
        
        # Step 3: For each predicted (Type2, Type3) combination, predict Type4
        for t3 in np.unique(t3_predictions['type3']):
            t3_str = str(t3)
            key = f"{t2_str}_{t3_str}"
            if key not in model_chain['type4']:
                continue
                
            # Get samples predicted as this (Type2, Type3) combination
            t3_mask = t2_mask & (type3_pred == t3)
            if np.sum(t3_mask) == 0:
                continue
                
            # Predict Type4 for these samples
            t4_predictions = model_chain['type4'][key].predict(X[t3_mask])
            type4_pred[t3_mask] = t4_predictions['type4']
    
    return {
        'type2': type2_pred,
        'type3': type3_pred,
        'type4': type4_pred
    }


class HierarchicalModelController:
    def __init__(self, data, df):
        self.data = data
        self.df = df
        self.models = {
            'type2': None,  # Base model for Type2 prediction
            'type3': {},    # Dictionary of Type3 models for each Type2 class
            'type4': {}     # Dictionary of Type4 models for each (Type2, Type3) combination
        }
        
    def train(self):
        """Train hierarchical models for Type2, Type3, and Type4."""
        if self.data.X_train is None:
            print("Skipping training due to insufficient data...")
            return
            
        # Train base Type2 model
        print("Training base Type2 model...")
        self.models['type2'] = RandomForest("Type2_Base", self.data.get_X_train(), self.data.df_train)
        self.models['type2'].train(self.data)
        
        # Get Type2 predictions for training data
        type2_train_pred = self.models['type2'].predict(self.data.get_X_train())
        
        # Train Type3 models for each Type2 class
        print("\nTraining Type3 models for each Type2 class...")
        unique_type2 = np.unique(self.data.get_y_train_type2())
        for t2 in unique_type2:
            # Filter data for this Type2 class
            mask = self.data.get_y_train_type2() == t2
            if np.sum(mask) < 3:  # Skip if too few samples
                continue
                
            # Create filtered data
            filtered_data = self._filter_data(mask)
            if filtered_data.X_train is None:
                continue
            
            # Train Type3 model
            t2_str = str(t2)
            model_name = f"Type3_{t2_str}"
            self.models['type3'][t2_str] = RandomForest(model_name, filtered_data.get_X_train(), filtered_data.df_train)
            self.models['type3'][t2_str].train(filtered_data)
            
            # Get Type3 predictions for training data
            type3_train_pred = self.models['type3'][t2_str].predict(filtered_data.get_X_train())
            
            # Train Type4 models for each (Type2, Type3) combination
            print(f"\nTraining Type4 models for Type2 class: {t2_str}")
            unique_type3 = np.unique(filtered_data.get_y_train_type3())
            for t3 in unique_type3:
                # Filter data for this (Type2, Type3) combination
                mask = filtered_data.get_y_train_type3() == t3
                if np.sum(mask) < 3:  # Skip if too few samples
                    continue
                    
                # Create filtered data
                filtered_data_t3 = self._filter_data_t3(filtered_data, mask)
                if filtered_data_t3.X_train is None:
                    continue
                
                # Train Type4 model
                t3_str = str(t3)
                key = f"{t2_str}_{t3_str}"
                model_name = f"Type4_{key}"
                self.models['type4'][key] = RandomForest(model_name, filtered_data_t3.get_X_train(), filtered_data_t3.df_train)
                self.models['type4'][key].train(filtered_data_t3)
    
    def predict(self, X_test):
        """Make predictions using the hierarchical models."""
        if self.models['type2'] is None:
            return {
                'type2': np.array([]),
                'type3': np.array([]),
                'type4': np.array([])
            }
            
        # Get predictions for all levels
        predictions = predict_hierarchical(self.models, X_test)
        
        # Ensure predictions match the test set size
        n_test = len(self.data.y_test_type2)
        if len(predictions['type2']) != n_test:
            # If predictions don't match test set size, return empty arrays
            return {
                'type2': np.array(['Unknown'] * n_test),
                'type3': np.array(['Unknown'] * n_test),
                'type4': np.array(['Unknown'] * n_test)
            }
            
        return predictions
    
    def _filter_data(self, mask):
        """Helper function to create filtered data object."""
        # Get the training data
        X_train = self.data.get_X_train()
        df_train = self.data.df_train
        
        # Create filtered data
        filtered_X = X_train[mask]
        filtered_df = df_train.iloc[mask].reset_index(drop=True)
        
        # Create new Data object
        filtered_data = Data(filtered_X, filtered_df)
        
        # Set train data to filtered data
        filtered_data.X_train = filtered_data.X
        filtered_data.df_train = filtered_data.df
        filtered_data.y_train_type2 = filtered_data.y2
        filtered_data.y_train_type3 = filtered_data.y3
        filtered_data.y_train_type4 = filtered_data.y4
        
        # Set test data to None since we don't need it for training
        filtered_data.X_test = None
        filtered_data.df_test = None
        filtered_data.y_test_type2 = None
        filtered_data.y_test_type3 = None
        filtered_data.y_test_type4 = None
        
        # Set valid classes
        filtered_data.classes_type2 = np.unique(filtered_data.y2)
        filtered_data.classes_type3 = np.unique(filtered_data.y3)
        filtered_data.classes_type4 = np.unique(filtered_data.y4)
        
        return filtered_data
        
    def _filter_data_t3(self, data, mask):
        """Helper function to create filtered data object for Type3."""
        # Get the training data
        X_train = data.get_X_train()
        df_train = data.df_train
        
        # Create filtered data
        filtered_X = X_train[mask]
        filtered_df = df_train.iloc[mask].reset_index(drop=True)
        
        # Create new Data object
        filtered_data = Data(filtered_X, filtered_df)
        
        # Set train data to filtered data
        filtered_data.X_train = filtered_data.X
        filtered_data.df_train = filtered_data.df
        filtered_data.y_train_type2 = filtered_data.y2
        filtered_data.y_train_type3 = filtered_data.y3
        filtered_data.y_train_type4 = filtered_data.y4
        
        # Set test data to None since we don't need it for training
        filtered_data.X_test = None
        filtered_data.df_test = None
        filtered_data.y_test_type2 = None
        filtered_data.y_test_type3 = None
        filtered_data.y_test_type4 = None
        
        # Set valid classes
        filtered_data.classes_type2 = np.unique(filtered_data.y2)
        filtered_data.classes_type3 = np.unique(filtered_data.y3)
        filtered_data.classes_type4 = np.unique(filtered_data.y4)
        
        return filtered_data


def model_predict(data, df, name):
    results = []
    print("RandomForest")
    
    # Initialize and train hierarchical model
    controller = HierarchicalModelController(data, df)
    controller.train()
    
    # Make predictions
    predictions = controller.predict(data.X_test)
    
    # Skip evaluation if any predictions are empty
    if len(predictions['type2']) == 0:
        print("Skipping evaluation due to insufficient predictions...")
        return
        
    # Create masks for each level
    type2_mask = predictions['type2'] != 'Unknown'
    type3_mask = (predictions['type2'] == data.y_test_type2) & (predictions['type3'] != 'Unknown')
    type4_mask = (predictions['type2'] == data.y_test_type2) & (predictions['type3'] == data.y_test_type3) & (predictions['type4'] != 'Unknown')
    
    if not any(type2_mask):
        print("All Type2 predictions are Unknown. Skipping evaluation...")
        return
        
    # Print results for each level
    print("\nType2 Classification Report:")
    print(classification_report(data.y_test_type2[type2_mask], predictions['type2'][type2_mask]))
    
    if any(type3_mask):
        print("\nType3 Classification Report (for correct Type2 predictions):")
        print(classification_report(data.y_test_type3[type3_mask], predictions['type3'][type3_mask]))
    else:
        print("\nNo valid Type3 predictions for evaluation.")
        
    # Evaluate Type4 predictions
    if np.any(type4_mask):
        print("\nType4 Classification Report:")
        print(classification_report(data.y_test_type4[type4_mask], predictions['type4'][type4_mask]))
    else:
        print("\nNo valid Type4 predictions for evaluation.")
        print("Type4 predictions were either 'Unknown' or parent predictions (Type2/Type3) were incorrect.")
        print("\nType4 Class Distribution in Test Set:")
        print(pd.Series(data.y_test_type4).value_counts())
        print("\nType4 Prediction Distribution:")
        print(pd.Series(predictions['type4']).value_counts())
    
    # Calculate chained classification score
    score = evaluate_chained_classification(
        y_true_type2=data.y_test_type2[type2_mask],
        y_true_type3=data.y_test_type3[type2_mask],
        y_true_type4=data.y_test_type4[type2_mask],
        y_pred_type2=predictions['type2'][type2_mask],
        y_pred_type3=predictions['type3'][type2_mask],
        y_pred_type4=predictions['type4'][type2_mask]
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