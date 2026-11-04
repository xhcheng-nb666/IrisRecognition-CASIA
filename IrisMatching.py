"""Regularized PCA -> Fisher LDA and nearest-center classifier."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


class IrisMatching:

    def __init__(self, n_components=100, shrinkage=0.1):
        self.n_components = n_components
        self.shrinkage = shrinkage

        # Seven rotation angles from Ma et al. (2003)
        self.angles = [-9, -6, -3, 0, 3, 6, 9]

        # Rotated class centers, populated separately
        self.rotated_centers = None

    def fit(self, X, y):
        self.labels = np.unique(y)

        # Standardization
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        # PCA
        self.pca = PCA(
            n_components=min(
                self.n_components,
                Xs.shape[0] - len(self.labels),
                Xs.shape[1]
            ),
            svd_solver='randomized',
            random_state=42
        ).fit(Xs)

        Z = self.pca.transform(Xs)

        # Fisher LDA
        self.lda = LinearDiscriminantAnalysis(
            solver='eigen',
            shrinkage=self.shrinkage,
            n_components=min(len(self.labels) - 1, Z.shape[1])
        )

        self.lda.fit(Z, y)

        F = self.lda.transform(Z)

        # Original class centers (without rotation)
        self.centers = np.stack([
            F[y == label].mean(axis=0)
            for label in self.labels
        ])

        self.rotated_centers = None

        return self

    def transform(self, X):
        return self.lda.transform(
            self.pca.transform(
                self.scaler.transform(X)
            )
        )

    # NEW: Generate centers for the seven rotations
    def fit_rotation(self, X_rotated, y_rotated, angles):
        """
        X_rotated: feature vectors from rotated training images
        y_rotated: class label for each feature vector
        angles: rotation angle for each feature vector

        All three arrays must have the same number of samples.
        """

        X_rotated = np.asarray(X_rotated)
        y_rotated = np.asarray(y_rotated)
        angles = np.asarray(angles)

        if not (
            len(X_rotated) == len(y_rotated) == len(angles)
        ):
            raise ValueError("Rotated data arrays must have equal lengths")

        # Apply the already-trained PCA and LDA
        F = self.transform(X_rotated)

        centers = []

        # Calculate one center per class and rotation angle
        for label in self.labels:

            class_centers = []

            for angle in self.angles:

                mask = (y_rotated == label) & (angles == angle)

                if not np.any(mask):
                    raise ValueError(
                        f"Missing training samples: class {label}, angle {angle}"
                    )

                center = F[mask].mean(axis=0)

                class_centers.append(center)

            centers.append(class_centers)

        # Shape: (108 classes, 7 rotations, LDA dimensions)
        self.rotated_centers = np.asarray(centers)

        return self

    def distances(self, X, metric='cosine', rotation=False):

        F = self.transform(X)

        if metric == 'L1':
            distance_metric = 'cityblock'

        elif metric == 'L2':
            distance_metric = 'euclidean'

        elif metric == 'cosine':
            distance_metric = 'cosine'

        else:
            raise ValueError(metric)

        # Original matching: 1 center per class
        if not rotation:
            return cdist(
                F,
                self.centers,
                metric=distance_metric
            )

        # NEW: Rotation-compensated matching
        if self.rotated_centers is None:
            raise ValueError(
                "Call fit_rotation() before rotation matching"
            )

        n_classes = len(self.labels)
        n_rotations = len(self.angles)

        # Flatten rotated centers for distance calculation
        centers = self.rotated_centers.reshape(
            -1,
            self.rotated_centers.shape[-1]
        )

        D = cdist(F, centers, metric=distance_metric)

        # Reshape into:
        # (test samples, classes, rotations)
        D = D.reshape(
            len(F),
            n_classes,
            n_rotations
        )

        # Choose minimum distance across seven rotations
        return D.min(axis=2)

    def predict(self, X, metric='cosine', rotation=False):

        D = self.distances(
            X,
            metric=metric,
            rotation=rotation
        )

        return self.labels[D.argmin(axis=1)]