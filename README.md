# K-MEANS-CLUSTERING-FOR-MNIST-DATASET-
# MNIST Clustering with K-Means and Evaluation

This repository contains code for clustering the MNIST handwritten digits dataset using K-Means and evaluating the results using various metrics.

## Project Overview

The goal of this project is to explore the application of K-Means clustering to the MNIST dataset and understand how well the resulting clusters align with the actual digit labels. We also evaluate the clustering performance using both unsupervised and supervised evaluation metrics.

**Important Note:** K-Means is an unsupervised learning algorithm. It does not directly predict digit labels. The evaluation performed here is to assess how well the clusters *correspond* to the true labels, not to measure K-Means's predictive accuracy.

## Code Description

The repository contains the following key steps:

1.  **Data Loading:** Loading the MNIST dataset using Keras.
2.  **Preprocessing:**
    * Flattening the 28x28 images into 784-element vectors.
    * Standard scaling the data using `StandardScaler`.
    * Applying Principal Component Analysis (PCA) to reduce dimensionality.
3.  **K-Means Clustering:** Applying K-Means with a specified number of clusters (e.g., 10 for the 10 digits).
4.  **Evaluation:**
    * **Unsupervised Metrics:**
        * Silhouette Score
        * Calinski-Harabasz Index
    * **Supervised-like Metrics (with caution):**
        * Accuracy, Precision, Recall (after mapping clusters to labels)
        * Adjusted Rand Index (ARI)
        * Homogeneity Score
        * Normalized Mutual Information (NMI)
    * `classification_report` (with the important caveats explained in the code).

## Requirements

* Python 3.x
* Libraries:
    * `keras`
    * `numpy`
    * `scikit-learn`
    * `pandas` (for simplified cluster-to-label mapping)

## Usage

1.  Clone the repository:
    ```bash
    git clone [repository URL]
    ```
2.  Install the required libraries:
    ```bash
    pip install keras numpy scikit-learn pandas
    ```
3.  Run the Python script (e.g., `mnist_kmeans.py`):
    ```bash
    python mnist_kmeans.py
    ```

## Interpretation of Results
* **ARI and Homogeneity:** These metrics show how well the clusters correspond to the labels, but they do not measure predictive accuracy.
* **Unsupervised Metrics:** Silhouette Score and Calinski-Harabasz Index provide insights into the quality of the clusters themselves.

**Important:** Do not interpret the "supervised-like" metrics as if they came from a supervised classification model. K-Means is not a classifier.

## Further Exploration

* Experiment with different values for `n_clusters` in K-Means.
* Try other clustering algorithms (e.g., Gaussian Mixture Models, DBSCAN).
* Explore different PCA component settings.
* Visualize the clusters using dimensionality reduction techniques (e.g., t-SNE).

## Author

Speranza Deejoe 

