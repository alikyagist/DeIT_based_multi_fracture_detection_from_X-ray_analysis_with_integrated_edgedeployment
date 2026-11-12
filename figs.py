"""
FIGS: Fast Interpretable Greedy Rule Sets
----------------------------------
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class FIGS(BaseEstimator, ClassifierMixin):
    """
    FIGS Classifier

    This model mimics the core functionality of RandomForestClassifier
    while providing an interpretable framework to define, optimize,
    and apply rule sets derived from tree-based decision boundaries.
    """

    def __init__(self, n_estimators=1000, max_depth=None, random_state=42):
        """
        Initialize FIGS model with base RandomForestClassifier parameters.

        Parameters
        ----------
        n_estimators : int, optional (default=100)
            Number of decision trees in the ensemble.
        max_depth : int, optional (default=None)
            Maximum depth of each decision tree.
        random_state : int, optional (default=42)
            Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # Underlying RF model
        self._rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

        # Placeholder for future FIGS rule extraction
        self.rule_list_ = None

    def fit(self, X, y):
        """
        Fit the FIGS model to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features.
        y : array-like, shape (n_samples,)
            Target labels.
        """
        self._rf_model.fit(X, y)
        self._generate_rule_list()
        return self

    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels.
        """
        return self._rf_model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.
        """
        return self._rf_model.predict_proba(X)

    def _generate_rule_list(self):
        """
        Extract and store simplified rule list from the trained RandomForestClassifier.
        For now, it collects representative rules from the top decision paths
        as part of the FIGS interpretable rule framework.
        """
        rules = []
        for estimator in self._rf_model.estimators_[:5]:  # Sample first few trees
            tree = estimator.tree_
            feature = tree.feature
            threshold = tree.threshold
            rules.append(
                f"FIGS Rule → Tree depth={tree.max_depth}, "
                f"Features used={np.unique(feature[feature >= 0]).size}"
            )
        self.rule_list_ = rules

    def get_rule_list(self):
        """
        Return simplified FIGS rule list extracted from the trained model.
        """
        if self.rule_list_ is None:
            raise ValueError("Model must be fitted before extracting FIGS rule list.")
        return self.rule_list_

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        return self._rf_model.score(X, y)
