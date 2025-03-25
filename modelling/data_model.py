import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
        # Store the input DataFrame and embeddings
        self.df = df
        self.embeddings = X
        
        # Get all type columns
        y2 = df['y2'].to_numpy()
        y3 = df['y3'].to_numpy()
        y4 = df['y4'].to_numpy()
        
        # Filter for Type2 classes with sufficient samples (>= 3)
        y2_series = pd.Series(y2)
        good_y2_value = y2_series.value_counts()[y2_series.value_counts() >= 3].index
        
        if len(good_y2_value) < 1:
            print("No Type2 classes have more than 3 records: Skipping ...")
            self.X_train = None
            return
            
        # Filter data for good Type2 classes
        mask = y2_series.isin(good_y2_value).values
        
        # Get filtered data
        y2_good = y2[mask]
        y3_good = y3[mask]
        y4_good = y4[mask]
        X_good = X[mask]
        df_good = df.iloc[mask].reset_index(drop=True)
        
        # Store filtered data
        self.X = X_good
        self.y2 = y2_good
        self.y3 = y3_good
        self.y4 = y4_good
        self.df = df_good
        
        # Split data
        train_idx, test_idx = train_test_split(
            np.arange(len(X_good)), test_size=0.2, random_state=0, stratify=y2_good)
        
        # Store train data
        self.X_train = X_good[train_idx]
        self.y_train_type2 = y2_good[train_idx]
        self.y_train_type3 = y3_good[train_idx]
        self.y_train_type4 = y4_good[train_idx]
        self.df_train = df_good.iloc[train_idx].reset_index(drop=True)
        
        # Store test data
        self.X_test = X_good[test_idx]
        self.y_test_type2 = y2_good[test_idx]
        self.y_test_type3 = y3_good[test_idx]
        self.y_test_type4 = y4_good[test_idx]
        self.df_test = df_good.iloc[test_idx].reset_index(drop=True)
        
        # Store valid classes
        self.classes_type2 = good_y2_value
        self.classes_type3 = np.unique(y3_good)  # Keep all Type3 classes
        self.classes_type4 = np.unique(y4_good)  # Keep all Type4 classes

    def get_X_train(self):
        return self.X_train
    def get_X_test(self):
        return self.X_test
    def get_y_train_type2(self):
        return self.y_train_type2
    def get_y_train_type3(self):
        return self.y_train_type3
    def get_y_train_type4(self):
        return self.y_train_type4
    def get_y_test_type2(self):
        return self.y_test_type2
    def get_y_test_type3(self):
        return self.y_test_type3
    def get_y_test_type4(self):
        return self.y_test_type4
    def get_embeddings(self):
        return self.embeddings

