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

        # Get all type columns
        y2 = df['y2'].to_numpy()
        y3 = df['y3'].to_numpy()
        y4 = df['y4'].to_numpy()
        
        # Filter for classes with sufficient samples (>= 3)
        y2_series = pd.Series(y2)
        y3_series = pd.Series(y3)
        y4_series = pd.Series(y4)
        
        good_y2_value = y2_series.value_counts()[y2_series.value_counts() >= 3].index
        good_y3_value = y3_series.value_counts()[y3_series.value_counts() >= 3].index
        good_y4_value = y4_series.value_counts()[y4_series.value_counts() >= 3].index

        if len(good_y2_value) < 1 or len(good_y3_value) < 1 or len(good_y4_value) < 1:
            print("None of the classes have more than 3 records: Skipping ...")
            self.X_train = None
            return

        # Filter data for good classes
        mask = (y2_series.isin(good_y2_value)) & (y3_series.isin(good_y3_value)) & (y4_series.isin(good_y4_value))
        y2_good = y2[mask]
        y3_good = y3[mask]
        y4_good = y4[mask]
        X_good = X[mask]

        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]

        # Split data for each type
        self.X_train, self.X_test, self.y_train_type2, self.y_test_type2 = train_test_split(
            X_good, y2_good, test_size=new_test_size, random_state=0, stratify=y2_good)
        _, _, self.y_train_type3, self.y_test_type3 = train_test_split(
            X_good, y3_good, test_size=new_test_size, random_state=0, stratify=y3_good)
        _, _, self.y_train_type4, self.y_test_type4 = train_test_split(
            X_good, y4_good, test_size=new_test_size, random_state=0, stratify=y4_good)
        
        self.embeddings = X
        self.classes_type2 = good_y2_value
        self.classes_type3 = good_y3_value
        self.classes_type4 = good_y4_value

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

