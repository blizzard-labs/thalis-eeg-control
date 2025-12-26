"""
Linear Discriminant Analysis classifier for finger motor imagery classification.

This module provides an LDA classifier with SVD solver that outputs probabilities
for three classes: Thumb, Index, and Pinky.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder


@dataclass
class ClassificationResult:
    """Result from a single classification."""
    predicted_class: str
    class_probabilities: Dict[str, float]
    raw_probabilities: np.ndarray
    
    def __repr__(self) -> str:
        probs_str = ", ".join(f"{k}: {v:.3f}" for k, v in self.class_probabilities.items())
        return f"Predicted: {self.predicted_class} | Probabilities: [{probs_str}]"


class CSPLDAClassifier:
    """
    Linear Discriminant Analysis classifier for CSP features.
    
    Uses SVD solver (suitable for any number of samples) and outputs
    probabilities for three finger classes: Thumb, Index, Pinky.
    
    Parameters
    ----------
    class_labels : List[str], optional
        Class labels in order. Default: ['Thumb', 'Index', 'Pinky']
    shrinkage : str or float, optional
        Shrinkage parameter. 'auto' uses Ledoit-Wolf lemma.
        Default: 'auto'
    store_covariance : bool, optional
        Whether to compute and store class covariances.
        Default: False
    
    Attributes
    ----------
    lda_ : LinearDiscriminantAnalysis
        Fitted sklearn LDA model
    label_encoder_ : LabelEncoder
        Encoder for string labels
    is_fitted_ : bool
        Whether the classifier has been fitted
    n_features_ : int
        Number of features after fitting
    """
    
    CLASS_LABELS = ['Thumb', 'Index', 'Pinky']
    
    def __init__(
        self,
        class_labels: Optional[List[str]] = None,
        shrinkage: Union[str, float] = 'auto',
        store_covariance: bool = False
    ):
        self.class_labels = class_labels or self.CLASS_LABELS
        self.shrinkage = shrinkage
        self.store_covariance = store_covariance
        
        # Use SVD solver (works with any n_samples, handles collinearity)
        # Note: SVD solver doesn't support shrinkage, so we use 'lsqr' if shrinkage is needed
        if shrinkage is not None and shrinkage != 'auto':
            self.lda_ = LinearDiscriminantAnalysis(
                solver='lsqr',
                shrinkage=shrinkage,
                store_covariance=store_covariance
            )
        else:
            # SVD solver is most robust but doesn't support shrinkage
            self.lda_ = LinearDiscriminantAnalysis(
                solver='svd',
                store_covariance=store_covariance
            )
        
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(self.class_labels)
        self.is_fitted_ = False
        self.n_features_ = None
        
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> 'CSPLDAClassifier':
        """
        Fit the LDA classifier.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features (CSP features, typically 24-dimensional)
        y : np.ndarray, shape (n_samples,)
            Class labels (can be strings or integers)
            
        Returns
        -------
        self : CSPLDAClassifier
            Fitted classifier
        """
        # Handle string labels
        if y.dtype == object or isinstance(y[0], str):
            y_encoded = self.label_encoder_.transform(y)
        else:
            y_encoded = y
            
        self.lda_.fit(X, y_encoded)
        self.is_fitted_ = True
        self.n_features_ = X.shape[1]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features) or (n_features,)
            Features to classify
            
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels as strings
        """
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
            
        X = self._ensure_2d(X)
        y_pred_encoded = self.lda_.predict(X)
        return self.label_encoder_.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features) or (n_features,)
            Features to classify
            
        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)
            Probability of each class (Thumb, Index, Pinky)
        """
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
            
        X = self._ensure_2d(X)
        return self.lda_.predict_proba(X)
    
    def classify(self, X: np.ndarray) -> ClassificationResult:
        """
        Classify a single sample and return detailed result.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_features,) or (1, n_features)
            Single sample features
            
        Returns
        -------
        result : ClassificationResult
            Classification result with predicted class and probabilities
        """
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
            
        X = self._ensure_2d(X)
        if X.shape[0] != 1:
            raise ValueError("classify() expects a single sample")
            
        proba = self.lda_.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        pred_class = self.class_labels[pred_idx]
        
        class_probs = {label: float(p) for label, p in zip(self.class_labels, proba)}
        
        return ClassificationResult(
            predicted_class=pred_class,
            class_probabilities=class_probs,
            raw_probabilities=proba
        )
    
    def classify_batch(self, X: np.ndarray) -> List[ClassificationResult]:
        """
        Classify multiple samples.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Multiple samples to classify
            
        Returns
        -------
        results : List[ClassificationResult]
            Classification results for each sample
        """
        X = self._ensure_2d(X)
        probas = self.predict_proba(X)
        
        results = []
        for proba in probas:
            pred_idx = np.argmax(proba)
            pred_class = self.class_labels[pred_idx]
            class_probs = {label: float(p) for label, p in zip(self.class_labels, proba)}
            results.append(ClassificationResult(
                predicted_class=pred_class,
                class_probabilities=class_probs,
                raw_probabilities=proba
            ))
        return results
    
    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        """Ensure X is 2D."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X
    
    def get_discriminant_values(self, X: np.ndarray) -> np.ndarray:
        """
        Get the discriminant function values.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Features
            
        Returns
        -------
        d : np.ndarray, shape (n_samples, n_components)
            Discriminant values (projection onto LDA axes)
        """
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        X = self._ensure_2d(X)
        return self.lda_.transform(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate classification accuracy.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test features
        y : np.ndarray, shape (n_samples,)
            True labels
            
        Returns
        -------
        accuracy : float
            Classification accuracy
        """
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
            
        # Handle string labels
        if y.dtype == object or isinstance(y[0], str):
            y_encoded = self.label_encoder_.transform(y)
        else:
            y_encoded = y
            
        X = self._ensure_2d(X)
        return self.lda_.score(X, y_encoded)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the classifier to disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the classifier
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'lda': self.lda_,
            'label_encoder': self.label_encoder_,
            'class_labels': self.class_labels,
            'is_fitted': self.is_fitted_,
            'n_features': self.n_features_,
            'shrinkage': self.shrinkage,
            'store_covariance': self.store_covariance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'CSPLDAClassifier':
        """
        Load a classifier from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the saved classifier
            
        Returns
        -------
        classifier : CSPLDAClassifier
            Loaded classifier
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        classifier = cls(
            class_labels=state['class_labels'],
            shrinkage=state['shrinkage'],
            store_covariance=state['store_covariance']
        )
        classifier.lda_ = state['lda']
        classifier.label_encoder_ = state['label_encoder']
        classifier.is_fitted_ = state['is_fitted']
        classifier.n_features_ = state['n_features']
        
        return classifier


class OnlineClassifier:
    """
    Wrapper for real-time classification with smoothing.
    
    Provides temporal smoothing of predictions using exponential
    moving average or majority voting.
    
    Parameters
    ----------
    classifier : CSPLDAClassifier
        Fitted classifier
    smoothing_method : str, optional
        'ema' for exponential moving average, 'majority' for voting.
        Default: 'ema'
    smoothing_alpha : float, optional
        EMA smoothing factor (0-1). Higher = more weight on recent.
        Default: 0.3
    window_size : int, optional
        Window size for majority voting.
        Default: 5
    """
    
    def __init__(
        self,
        classifier: CSPLDAClassifier,
        smoothing_method: str = 'ema',
        smoothing_alpha: float = 0.3,
        window_size: int = 5
    ):
        self.classifier = classifier
        self.smoothing_method = smoothing_method
        self.smoothing_alpha = smoothing_alpha
        self.window_size = window_size
        
        # State for smoothing
        self._smoothed_proba = None
        self._prediction_history = []
        
    def reset(self) -> None:
        """Reset smoothing state."""
        self._smoothed_proba = None
        self._prediction_history = []
        
    def update(self, features: np.ndarray) -> ClassificationResult:
        """
        Classify features and update smoothing state.
        
        Parameters
        ----------
        features : np.ndarray, shape (n_features,)
            Single sample features
            
        Returns
        -------
        result : ClassificationResult
            Smoothed classification result
        """
        # Get raw probabilities
        raw_proba = self.classifier.predict_proba(features.reshape(1, -1))[0]
        
        if self.smoothing_method == 'ema':
            return self._update_ema(raw_proba)
        else:
            return self._update_majority(raw_proba)
    
    def _update_ema(self, raw_proba: np.ndarray) -> ClassificationResult:
        """Update using exponential moving average."""
        if self._smoothed_proba is None:
            self._smoothed_proba = raw_proba.copy()
        else:
            self._smoothed_proba = (
                self.smoothing_alpha * raw_proba + 
                (1 - self.smoothing_alpha) * self._smoothed_proba
            )
        
        # Renormalize
        smoothed = self._smoothed_proba / self._smoothed_proba.sum()
        
        pred_idx = np.argmax(smoothed)
        pred_class = self.classifier.class_labels[pred_idx]
        class_probs = {
            label: float(p) 
            for label, p in zip(self.classifier.class_labels, smoothed)
        }
        
        return ClassificationResult(
            predicted_class=pred_class,
            class_probabilities=class_probs,
            raw_probabilities=smoothed
        )
    
    def _update_majority(self, raw_proba: np.ndarray) -> ClassificationResult:
        """Update using majority voting."""
        pred_idx = np.argmax(raw_proba)
        self._prediction_history.append(pred_idx)
        
        # Keep only recent predictions
        if len(self._prediction_history) > self.window_size:
            self._prediction_history.pop(0)
        
        # Majority vote
        votes = np.bincount(self._prediction_history, minlength=len(self.classifier.class_labels))
        majority_idx = np.argmax(votes)
        
        # Estimate probabilities from vote distribution
        vote_proba = votes / votes.sum()
        
        pred_class = self.classifier.class_labels[majority_idx]
        class_probs = {
            label: float(p) 
            for label, p in zip(self.classifier.class_labels, vote_proba)
        }
        
        return ClassificationResult(
            predicted_class=pred_class,
            class_probabilities=class_probs,
            raw_probabilities=vote_proba
        )


# =============================================================================
# Demo / Test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LDA Classifier Demo")
    print("=" * 60)
    
    # Generate synthetic data mimicking CSP features
    np.random.seed(42)
    n_samples_per_class = 50
    n_features = 24  # Dual-band CSP features
    
    # Create class-specific distributions
    # Thumb: higher values in first few features
    X_thumb = np.random.randn(n_samples_per_class, n_features)
    X_thumb[:, :8] += 1.5
    
    # Index: higher values in middle features  
    X_index = np.random.randn(n_samples_per_class, n_features)
    X_index[:, 8:16] += 1.5
    
    # Pinky: higher values in last features
    X_pinky = np.random.randn(n_samples_per_class, n_features)
    X_pinky[:, 16:] += 1.5
    
    # Combine
    X = np.vstack([X_thumb, X_index, X_pinky])
    y = np.array(['Thumb'] * n_samples_per_class + 
                 ['Index'] * n_samples_per_class + 
                 ['Pinky'] * n_samples_per_class)
    
    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    
    # Split train/test
    n_train = int(0.8 * len(y))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"\nTraining samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {n_features}")
    
    # Create and fit classifier
    classifier = CSPLDAClassifier()
    classifier.fit(X_train, y_train)
    
    # Test accuracy
    accuracy = classifier.score(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.2%}")
    
    # Single sample classification
    print("\n--- Single Sample Classification ---")
    test_sample = X_test[0]
    result = classifier.classify(test_sample)
    print(f"True label: {y_test[0]}")
    print(f"{result}")
    
    # Batch classification
    print("\n--- Batch Classification (first 5) ---")
    results = classifier.classify_batch(X_test[:5])
    for i, (res, true_label) in enumerate(zip(results, y_test[:5])):
        correct = "✓" if res.predicted_class == true_label else "✗"
        print(f"  {i+1}. {res} (True: {true_label}) {correct}")
    
    # Test save/load
    print("\n--- Save/Load Test ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "classifier.pkl"
        classifier.save(save_path)
        print(f"Saved to: {save_path}")
        
        loaded = CSPLDAClassifier.load(save_path)
        loaded_acc = loaded.score(X_test, y_test)
        print(f"Loaded classifier accuracy: {loaded_acc:.2%}")
    
    # Online classifier demo
    print("\n--- Online Classifier (EMA Smoothing) ---")
    online = OnlineClassifier(classifier, smoothing_method='ema', smoothing_alpha=0.3)
    
    # Simulate streaming predictions
    for i in range(5):
        result = online.update(X_test[i])
        print(f"  Window {i+1}: {result}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
