from model.randomforest import RandomForest
import numpy as np


def model_predict(data, df, name):
    results = []
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)


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