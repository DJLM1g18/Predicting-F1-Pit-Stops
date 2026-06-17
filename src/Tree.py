import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score
import math
import numpy as np
from src import Misc

# Function to label encode user-specified
# columns in a pandas df
def label_encode(df, cols):
    # We return a modified df,
    # not modify the original
    # (keeping it in the theme
    # of how pandas-type functions
    # usually work)
    df_return = df.copy()
    # We return a list of label encoders
    # for each of the columns
    label_encoders = []
    for column in cols:
        # Define a new label encoder,
        # fit it to the next column,
        # transform the df
        le = LabelEncoder()
        le.fit(df[column])
        df_return[column] = le.transform(df[column])
        label_encoders.append(le)
    return df_return,label_encoders

# XGBoost model
def xgboost(df,features,target,target_recallprecision):
    # Separate features and target
    X = df[features]
    y = df[target]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create model
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic"
    )
    
    # Fit model
    model.fit(X_train, y_train)
    
    # And now we start grabbing recall-precision values
    # To start, we need the raw probabilities our model gives
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Now do the recall-precision tradeoff stuff
    recall_list,precision_list, accuracy_list, optimal_threshold, closest_recallprecision, closest_accuracy = Misc.recall_precision_tradeoff(y_prob, y_test, target_recallprecision)
    
    return model, recall_list,precision_list, accuracy_list, optimal_threshold, closest_recallprecision, closest_accuracy,X_test,y_test