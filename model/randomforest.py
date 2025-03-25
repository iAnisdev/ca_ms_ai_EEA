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
        
        # Initialize separate models for each type
        self.model_type2 = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.model_type3 = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.model_type4 = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        
        # Store predictions for each type
        self.predictions_type2 = None
        self.predictions_type3 = None
        self.predictions_type4 = None
        
        self.data_transform()

    def train(self, data) -> None:
        # Train Type2 model
        self.model_type2.fit(data.X_train, data.y_train_type2)
        
        # Train Type3 model using Type2 predictions
        type2_train_pred = self.model_type2.predict(data.X_train)
        self.model_type3.fit(data.X_train, data.y_train_type3)
        
        # Train Type4 model using Type2 and Type3 predictions
        type3_train_pred = self.model_type3.predict(data.X_train)
        self.model_type4.fit(data.X_train, data.y_train_type4)

    def predict(self, X_test: pd.Series):
        # Predict Type2
        self.predictions_type2 = self.model_type2.predict(X_test)
        
        # Predict Type3
        self.predictions_type3 = self.model_type3.predict(X_test)
        
        # Predict Type4
        self.predictions_type4 = self.model_type4.predict(X_test)
        
        return {
            'type2': self.predictions_type2,
            'type3': self.predictions_type3,
            'type4': self.predictions_type4
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

