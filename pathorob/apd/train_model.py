import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def fit_and_evaluate(args):
    """
    Trains LR model for given c and evaluates performance on validation set.

    Args: 
        args tuple: c, train_x, train_y, val_x, val_y, eval_func.

    Returns:
        c: Specified regularization parameter.
        val_score: Performance score for validation set.
    """
    c, train_x, train_y, val_x, val_y, eval_func = args
    model = LogisticRegression(C=c, random_state=0)
    model.fit(train_x, train_y)
    val_score = eval_func(val_y, model.predict(val_x))
    return c, val_score


def train_logistic_regression(train_x, train_y, val_x, val_y, test_xs, test_ys, eval_func=balanced_accuracy_score):
    """
    Trains LR model using the optimal regularization parameter c selected from the validation set.

    Args:
        train_x (tuple with patch features), train_y (tuple with integer labels): Training set.
        val_x (tuple with patch features), val_y (tuple with integer labels): Validation set.
        test_xs (list with feature tuples), test_ys (list with label tuples): Test sets. 
        eval_func: Evaluation metric (default=balanced_accuracy_score).

    Returns:
        final_model: Trained LR model.
        best_c: Optimal regularization parameter c selected from the validation set.
        test_scores (list): Performance scores for each test set.
    """
    # Generate C values
    C_POWER_RANGE = np.linspace(-8, 4, 15)
    Cs = 10**C_POWER_RANGE
    
    # Grid search
    args_list = [(c, train_x, train_y, val_x, val_y, eval_func) for c in Cs]
    results = list(map(fit_and_evaluate, args_list))
    
    # Find best C
    best_c, best_score = max(results, key=lambda x: x[1])
    
    # Train final model with best C and evaluate on test sets
    final_model = LogisticRegression(C=best_c)
    final_model.fit(train_x, train_y)
    test_scores = [eval_func(test_y, final_model.predict(test_x)) for test_x, test_y in zip(test_xs, test_ys)]
    
    return final_model, best_c, test_scores
