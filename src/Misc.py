from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score
import numpy as np
import math
import matplotlib.pyplot as plt

# This function takes in the output of a model, the true
# classification values, and a target recall-precision that
# the user wants to reach.
# It then ranges through threshold values, and for each
# threshold records the recall and precision for that
# threshold. We find the threshold that produces the
# closest possible recall-precision to the user-defined
# target (in Euclidean distance).
def recall_precision_tradeoff(y_prob, y_test, target_recallprecision):
    # Defining how many thresholds we sweep
    thresholds = np.linspace(0,1,100)
    # Lists to keep track of all the recall,precision and accuracy values attained
    recall_list = []
    precision_list = []
    accuracy_list = []
    # Variables that keep track of how close
    # we get to the target recall-precision balance
    best_distance = math.inf
    optimal_threshold = -math.inf
    closest_recallprecision = (-math.inf,-math.inf)
    closest_accuracy = -math.inf
    for threshold in thresholds:
        # Convert raw probabilities to decisions based on the current threshold
        y_pred = (y_prob >= threshold).astype(int)
        # Get the performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred,zero_division=0)
        recall = recall_score(y_test, y_pred)
        # Add them to the lists
        recall_list.append(recall)
        precision_list.append(precision)
        accuracy_list.append(accuracy)
        # ... and now check if the current recall-precision balance is closer to what
        # the user wanted (under Euclidean distance)
        distance = math.sqrt((target_recallprecision[0] - recall) **2 + (target_recallprecision[1] - precision) ** 2)
        if (distance <= best_distance):
            optimal_threshold = threshold
            closest_recallprecision = (recall,precision)
            closest_accuracy = accuracy
            best_distance = distance
    return recall_list,precision_list, accuracy_list, optimal_threshold, closest_recallprecision, closest_accuracy

# This function takes in two pandas df's
# (D a dataset, S a subset of D), and computes the
# lift for col == value
# Returns -1 if baserate == 0
def lift(S,D,col,value):
    # First, compute the base rate established by D
    baserate = len(D[D[col[0]] == value]) / (len(D))
    # Make sure we don't divide by 0
    if (baserate == 0):
        return -1
    # Now compute the rate in S
    Srate = len(S[S[col[0]] == value]) / (len(S))
    # ... and return the lift
    return (Srate / baserate)

# Empirically establish a 99% null
# interval for lift (two-tailed)
# D is the parent dataset,
# N is the size to sample,
# col == value is the property of interest,
# experiments controls how many lift samples to compute,
# dp controls the decimal point to round to (default is 3dp.)
def empirical_lift_NI(D, N, col, value, experiments=1000, dp=3):
    lifts = []
    for _ in range(0,experiments):
        # Produce the sample
        S = D.sample(N)
        # Append the lift
        lifts.append(lift(S,D,col,value))
    # Now get the quantiles
    lifts = np.array(lifts)
    lower = np.quantile(lifts,0.005)
    upper = np.quantile(lifts,0.995)
    return (round(float(lower), dp), round(float(upper), dp))

# This function takes in a list of categories (x),
# their lift values (y), the 99% null
# interval estimates (y_lower,y_upper), and plots it
def lift_plot(x,y_lower,y_upper,y):
    for c, l, u, obs in zip(x, y_lower, y_upper, y):
        plt.plot([c, c], [l, u], lw=5)
        plt.scatter(c, obs, color="red", s=100, edgecolor="black", zorder=2)

    plt.axhline(1, ls="--", color="black")
    plt.ylabel("Lift")
    plt.show()