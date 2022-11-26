"""Use various methods to model data and record performance.

Author: Jake Myers

Note: This version excludes PCA from runtime and memory recording.
"""

import time
import tracemalloc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from warnings import simplefilter


# sklearn kept giving me a FutureWarning about column names being ints
# https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
simplefilter(action='ignore', category=FutureWarning)
NUMBER_OF_RUNS = 100


# Linear regression done, score returned.
def lm_fit_score(X_train, X_test, y_train, y_test):
    lm = LinearRegression()

    X_train_arr = np.array(X_train)
    y_train_arr = np.array(y_train).reshape(-1, 1)

    lm_reg = lm.fit(X_train_arr, y_train_arr)
    return lm_reg.score(X_test, y_test)


# Decision tree regression done, score returned.
def dt_fit_score(X_train, X_test, y_train, y_test):
    dt = DecisionTreeRegressor(max_leaf_nodes=50)

    X_train_arr = np.array(X_train)
    y_train_arr = np.array(y_train).reshape(-1, 1)

    dt_reg = dt.fit(X_train_arr, y_train_arr)
    return dt_reg.score(X_test,y_test)


# Tries different values of k to find elbow.
# Given an upper k limit.
def find_kmeans_k(upper_limit_k: int):
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X) for k in range(1, upper_limit_k)]
    inertias = [model.inertia_ for model in kmeans_per_k]

    plt.figure()
    plt.plot(inertias, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Inertia", fontsize=14)
    plt.savefig("ElbowSearch")


def kmeans_get_ypred(X, k: int):
    kmeans = KMeans(n_clusters=k)  # Random is random (different each time)
    return kmeans.fit_predict(X)


# Use PCA to get the n most important columns. Returns new dataframe.
# Reference: https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
def get_f_important_features(X, f: int, n_components: float):
    pca = PCA(n_components=n_components)
    pca_model = pca.fit_transform(X)

    # Number of components
    n_pcs = pca.components_.shape[0]
    # Index of most important feature in each component
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    # Feature names
    initial_feature_names = [col for col in X.columns]
    # Putting feature names in order of importance
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

    # Uses the first f column names (most important) to build a new dataframe.
    return X[most_important_names[:f]]


if __name__ == '__main__':
    tracemalloc.start()
    x_digits, y_digits = load_digits(return_X_y=True)
    X = pd.DataFrame(x_digits)
    y = pd.DataFrame(y_digits)

    # ORIGINAL
    # Linear Regression
    lm_orig_score = 0
    lm_orig_mem = 0
    tic = time.perf_counter()   # Start timestamp
    for i in range(NUMBER_OF_RUNS):
        tracemalloc.reset_peak()    # Reset memory tracking
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        lm_orig_score += lm_fit_score(X_train, X_test, y_train, y_test)
        lm_orig_mem += tracemalloc.get_traced_memory()[0]   # Add memory peak to sum
    toc = time.perf_counter()   # End timestamp
    lm_orig_time = (toc - tic)/NUMBER_OF_RUNS  # Avg runtime of each
    lm_orig_mem /= NUMBER_OF_RUNS   # Avg memory usage
    lm_orig_score /= NUMBER_OF_RUNS

    # Decision Tree Regression
    dt_orig_score = 0
    dt_orig_mem = 0
    tic = time.perf_counter()  # Start timestamp
    for i in range(NUMBER_OF_RUNS):
        tracemalloc.reset_peak()  # Reset memory tracking
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        dt_orig_score += dt_fit_score(X_train, X_test, y_train, y_test)
        dt_orig_mem += tracemalloc.get_traced_memory()[0]  # Add memory peak to sum
    toc = time.perf_counter()  # End timestamp
    dt_orig_time = (toc - tic) / NUMBER_OF_RUNS  # Avg runtime of each
    dt_orig_mem /= NUMBER_OF_RUNS  # Avg memory usage
    dt_orig_score /= NUMBER_OF_RUNS
    print('Done processing original')


    # PCA
    # PCA reduction before recording
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # A new dataframe with 10 most important columns (from 64)
    # Uses an accuracy of 0.95 to make components.
    X_important = get_f_important_features(X_train, 10, 0.95)
    X_important_cols = [name for name in X_important.columns]
    X_train, X_test, y_train, y_test = train_test_split(X[X_important_cols], y, test_size=0.2)

    # Linear Regression
    lm_pca_score = 0
    lm_pca_mem = 0
    tic = time.perf_counter()  # Start timestamp
    for i in range(NUMBER_OF_RUNS):
        tracemalloc.reset_peak()  # Reset memory tracking
        lm_pca_score += lm_fit_score(X_train, X_test, y_train, y_test)
        lm_pca_mem += tracemalloc.get_traced_memory()[0]  # Add memory peak to sum
    toc = time.perf_counter()  # End timestamp
    lm_pca_time = (toc - tic) / NUMBER_OF_RUNS  # Avg runtime of each
    lm_pca_mem /= NUMBER_OF_RUNS  # Avg memory usage
    lm_pca_score /= NUMBER_OF_RUNS

    # Decision Tree Regression
    dt_pca_score = 0
    dt_pca_mem = 0
    tic = time.perf_counter()  # Start timestamp
    for i in range(NUMBER_OF_RUNS):
        tracemalloc.reset_peak()  # Reset memory tracking
        dt_pca_score += dt_fit_score(X_train, X_test, y_train, y_test)
        dt_pca_mem += tracemalloc.get_traced_memory()[0]  # Add memory peak to sum
    toc = time.perf_counter()  # End timestamp
    dt_pca_time = (toc - tic) / NUMBER_OF_RUNS  # Avg runtime of each
    dt_pca_mem /= NUMBER_OF_RUNS  # Avg memory usage
    dt_pca_score /= NUMBER_OF_RUNS
    print('Done processing PCA')


    # CLUSTERING WITH KMEANS
    # What is k?
    #find_kmeans_k(20)
    # 9 on lower side, 16 on higher side (See ElbowSearch.png).

    # Linear Regression
    lm_kmeans_score = 0
    lm_kmeans_mem = 0
    tic = time.perf_counter()  # Start timestamp
    for i in range(NUMBER_OF_RUNS):
        tracemalloc.reset_peak()  # Reset memory tracking
        # Copies X, makes a new column in it, the cluster assignment of each row in X
        X_with_cluster = X.copy(deep=True)
        X_with_cluster['cluster assignment'] = kmeans_get_ypred(X, 9)
        # Redefine Train and Test data with new feature column ('cluster assignment').
        X_train, X_test, y_train, y_test = train_test_split(X_with_cluster, y, test_size=0.2)
        lm_kmeans_score += lm_fit_score(X_train, X_test, y_train, y_test)
        lm_kmeans_mem += tracemalloc.get_traced_memory()[0]  # Add memory peak to sum
    toc = time.perf_counter()  # End timestamp
    lm_kmeans_time = (toc - tic) / NUMBER_OF_RUNS  # Avg runtime of each
    lm_kmeans_mem /= NUMBER_OF_RUNS  # Avg memory usage
    lm_kmeans_score /= NUMBER_OF_RUNS

    # Decision Tree Regression
    dt_kmeans_score = 0
    dt_kmeans_mem = 0
    tic = time.perf_counter()  # Start timestamp
    for i in range(NUMBER_OF_RUNS):
        tracemalloc.reset_peak()  # Reset memory tracking
        # Copies X, makes a new column in it, the cluster assignment of each row in X
        X_with_cluster = X.copy(deep=True)
        X_with_cluster['cluster assignment'] = kmeans_get_ypred(X, 9)
        # Redefine Train and Test data with new feature column ('cluster assignment').
        X_train, X_test, y_train, y_test = train_test_split(X_with_cluster, y, test_size=0.2)
        dt_kmeans_score += dt_fit_score(X_train, X_test, y_train, y_test)
        dt_kmeans_mem += tracemalloc.get_traced_memory()[0]  # Add memory peak to sum
    toc = time.perf_counter()  # End timestamp
    dt_kmeans_time = (toc - tic) / NUMBER_OF_RUNS  # Avg runtime of each
    dt_kmeans_mem /= NUMBER_OF_RUNS  # Avg memory usage
    dt_kmeans_score /= NUMBER_OF_RUNS
    print('Done processing KMeans')


    # KMEANS PLUS PCA
    # PCA reduction before recording
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # A new dataframe with 10 most important columns (from 64)
    # Uses an accuracy of 0.95 to make components.
    X_important = get_f_important_features(X_train, 10, 0.95)
    X_important_cols = [name for name in X_important.columns]

    # Linear Regression
    lm_combo_score = 0
    lm_combo_mem = 0
    tic = time.perf_counter()  # Start timestamp
    for i in range(NUMBER_OF_RUNS):
        tracemalloc.reset_peak()  # Reset memory tracking

        # Copies X, makes a new column in it, the cluster assignment of each row in X
        X_with_cluster = X.copy(deep=True)
        X_with_cluster['cluster assignment'] = kmeans_get_ypred(X, 9)

        X_train, X_test, y_train, y_test = train_test_split(X_with_cluster[X_important_cols], y, test_size=0.2)

        lm_combo_score += lm_fit_score(X_train, X_test, y_train, y_test)
        lm_combo_mem += tracemalloc.get_traced_memory()[0]  # Add memory peak to sum
    toc = time.perf_counter()  # End timestamp
    lm_combo_time = (toc - tic) / NUMBER_OF_RUNS  # Avg runtime of each
    lm_combo_mem /= NUMBER_OF_RUNS  # Avg memory usage
    lm_combo_score /= NUMBER_OF_RUNS

    # Decision Tree Regression
    dt_combo_score = 0
    dt_combo_mem = 0
    tic = time.perf_counter()  # Start timestamp
    for i in range(NUMBER_OF_RUNS):
        tracemalloc.reset_peak()  # Reset memory tracking

        # Copies X, makes a new column in it, the cluster assignment of each row in X
        X_with_cluster = X.copy(deep=True)
        X_with_cluster['cluster assignment'] = kmeans_get_ypred(X, 9)

        X_train, X_test, y_train, y_test = train_test_split(X_with_cluster[X_important_cols], y, test_size=0.2)

        dt_combo_score += dt_fit_score(X_train, X_test, y_train, y_test)
        dt_combo_mem += tracemalloc.get_traced_memory()[0]  # Add memory peak to sum
    toc = time.perf_counter()  # End timestamp
    dt_combo_time = (toc - tic) / NUMBER_OF_RUNS  # Avg runtime of each
    dt_combo_mem /= NUMBER_OF_RUNS  # Avg memory usage
    dt_combo_score /= NUMBER_OF_RUNS
    print('Done processing KMeans + PCA')


    # OUTPUT
    print(f'\nLR = Linear Regression, DT = Decision Tree Regression')
    print(f'LR Score (Original): {round(lm_orig_score, 5)}, AVG Runtime: {round(lm_orig_time, 5)} Seconds, AVG Memory Usage: {lm_orig_mem} Bytes')
    print(f'DT Score (Original): {round(dt_orig_score, 5)}, AVG Runtime: {round(dt_orig_time, 5)} Seconds, AVG Memory Usage: {dt_orig_mem} Bytes\n')
    print(f'LR Score (PCA Reduced): {round(lm_pca_score, 5)}, AVG Runtime: {round(lm_pca_time, 5)} Seconds, AVG Memory Usage: {lm_pca_mem} Bytes')
    print(f'DT Score (PCA Reduced): {round(dt_pca_score, 5)}, AVG Runtime: {round(dt_pca_time, 5)} Seconds, AVG Memory Usage: {dt_pca_mem} Bytes\n')
    print(f'LR Score (KMeans): {round(lm_kmeans_score, 5)}, AVG Runtime: {round(lm_kmeans_time, 5)} Seconds, AVG Memory Usage: {lm_kmeans_mem} Bytes')
    print(f'DT Score (KMeans): {round(dt_kmeans_score, 5)}, AVG Runtime: {round(dt_kmeans_time, 5)} Seconds, AVG Memory Usage: {dt_kmeans_mem} Bytes\n')
    print(f'LR Score (PCA + KMeans): {round(lm_combo_score, 5)}, AVG Runtime: {round(lm_combo_time, 5)} Seconds, AVG Memory Usage: {lm_combo_mem} Bytes')
    print(f'DT Score (PCA + KMeans): {round(dt_combo_score, 5)}, AVG Runtime: {round(dt_combo_time, 5)} Seconds, AVG Memory Usage: {dt_combo_mem} Bytes')
