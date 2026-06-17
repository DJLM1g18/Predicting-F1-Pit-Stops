import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src import Misc

# This function trains one global logisitic regression model
# on the dataset, using one-hot encoding on the categorical features.
# Arguments: `df` is the dataframe of the dataset
# `categorical_cols` contains the categorical columns to encode
# `numeric_cols` contains the numerical columns
# `target_recallprecision` allows the user to define a
# target recall-precision balance they'd like to target,
# and we find the closest attainable recall-precision
# (using Euclidean distance)
def global_onehot(df, categorical_cols, numeric_cols, target_recallprecision):
    # We start by splittng the features and target
    X = df.drop(columns=["PitNextLap"])
    y = df["PitNextLap"]
    # We define a ColumnTransformer object specifying the preprocessing steps.
    # For categorical variables, we apply OneHotEncoder with `drop="first"` to
    # avoid perfect multicollinearity in the encoded feature space. Any unseen
    # categories encountered at inference time are handled via `handle_unknown="ignore"`,
    # meaning they are encoded as all zeros (i.e. the absence of all known categories).
    #
    # For numerical features, we apply StandardScaler, which standardises each feature
    # to have mean 0 and variance 1. This ensures all features operate on a comparable
    # scale, preventing features with larger magnitudes from disproportionately
    # influencing the gradient updates in logistic regression.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first",handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ],
        remainder="passthrough"
    )
    
    # We use `class_weight="balanced"` since we already have seen
    # our data is not perfectly balanced (roughly 20% of the entries
    # give "PitNextLap" == 1).
    #
    # The other hyperparameters were tuned for balanced performance
    model = LogisticRegression(
        tol=1e-3,
        max_iter=1000,
        C=0.5,
        solver="saga",
        class_weight="balanced"
    )
    
    # Given data, our pipeline is:
    # first preprocess it (one-hot encode + scaling),
    # then use the model.
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # Produce our test-train split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Learn
    clf.fit(X_train, y_train)
    
    # And now we start grabbing recall-precision values
    # To start, we need the raw probabilities our model gives
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Now do the recall-precision tradeoff stuff
    recall_list,precision_list, accuracy_list, optimal_threshold, closest_recallprecision, closest_accuracy = Misc.recall_precision_tradeoff(y_prob, y_test, target_recallprecision)

    return clf,recall_list,precision_list,accuracy_list,optimal_threshold,closest_recallprecision,closest_accuracy