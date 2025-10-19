import numpy as np
import time

from sklearn.neighbors import KNeighborsRegressor


class FastLOOCV:
    """
    A class to perform Fast Leave-One-Out Cross-Validation (LOOCV).

    Attributes
    ----------
    data : The dataset used for performing LOOCV. Typically this includes features
           and labels required for model training and evaluation.
    """

    def __init__(self, data):
        """
        Initialize the FastLOOCV object.

        Parameters
        ----------
        data : array-like
            Input dataset to be used for LOOCV.
        """
        self.data = data

    def do_fast_loocv(self, k_values, sample_size=None):
        """
        Perform fast leave-one-out cross-validation for a set of K values.

        Parameters
        ----------
        k_values : array-like
            List or array of integer K values to evaluate (e.g., number of neighbors).
        sample_size : int, optional
            Number of samples to randomly select from the training set.
            If None, all available samples are used.

        Returns
        -------
        score : numpy.ndarray
            Vector containing performance scores for each K in k_values.
        elapsed_time : float
            Execution time (in seconds) for running the procedure.
        """
        start_time = time.time()

        if sample_size is not None and sample_size < len(self.data):
            indices = np.random.choice(len(self.data), size=sample_size, replace=False)
            data_subset = self.data[indices]
        else:
            data_subset = self.data
        
        # Asume data_subset is structured like a tuple (X, y) where X are features
        # and y are labels.
        X, y = data_subset[:, :-1], data_subset[:, -1]

        scores = np.zeros(len(k_values))
        
        # For each k, perform leave-one-out CV
        for i, k in enumerate(k_values):
            # Fit k-NN and predict on the same data
            knn = KNeighborsRegressor(n_neighbors=k + 1, algorithm='kd_tree')
            knn.fit(X, y)
            y_predict_fast = knn.predict(X)

            # Compute the adjusted MSE for LOOCV
            errors = (y - y_predict_fast) ** 2
            mse = np.mean(errors)
            scaling_factor = ((k+1)**2 / k**2)
            scores[i] = mse * scaling_factor

        elapsed_time = time.time() - start_time
        return scores, elapsed_time
        
    def do_normal_loocv(self, k_values, sample_size=None):
        """
        Perform standard leave-one-out cross-validation for a set of K values.

        Parameters
        ----------
        k_values : array-like
            List or array of integer K values to evaluate (e.g., number of neighbors).
        sample_size : int, optional
            Number of samples to randomly select from the training set.
            If None, all available samples are used.

        Returns
        -------
        score : numpy.ndarray
            Vector containing performance scores for each K in k_values.
        elapsed_time : float
            Execution time (in seconds) for running the procedure.
        """
        start_time = time.time()

        if sample_size is not None and sample_size < len(self.data):
            indices = np.random.choice(len(self.data), size=sample_size, replace=False)
            data_subset = self.data[indices]
        else:
            data_subset = self.data

        X, y = data_subset[:, :-1], data_subset[:, -1]
        n = len(y)

        scores = np.zeros(len(k_values))

        # For each k, perform leave-one-out CV
        for i, k in enumerate(k_values):
            errors = np.zeros(n)
            for j in range(n):
                # Leave one out
                X_train = np.delete(X, j, axis=0)
                y_train = np.delete(y, j)
                X_val = X[j].reshape(1, -1)
                y_val = y[j]

                # Fit k-NN and predict
                knn = KNeighborsRegressor(n_neighbors=k, algorithm='kd_tree')  
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_val)[0] 
                errors[j] = (y_pred - y_val) ** 2

            scores[i] = np.mean(errors)

        elapsed_time = time.time() - start_time
        return scores, elapsed_time
