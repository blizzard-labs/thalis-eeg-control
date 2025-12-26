"""
Pairwise Common Spatial Pattern (CSP) Feature Extraction Module.

Implements one-vs-one pairwise CSP for three-class motor imagery classification:
- Thumb vs Index
- Index vs Pinky  
- Thumb vs Pinky

CSP finds spatial filters that maximize variance for one class while
minimizing variance for another class. For three classes, we use
pairwise binary CSP rather than multiclass CSP.

Uses MNE-Python for CSP implementation.

References:
    Blankertz et al., "Optimizing Spatial filters for Robust EEG 
    Single-Trial Analysis", IEEE Signal Processing Magazine, 2008.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import threading

# MNE for CSP
from mne.decoding import CSP

# Sklearn for optional downstream classification
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# Class labels for finger motor imagery
class FingerLabels:
    """Class labels for finger motor imagery tasks."""
    THUMB = 0
    INDEX = 1
    PINKY = 2
    
    @classmethod
    def get_name(cls, label: int) -> str:
        """Get human-readable name for label."""
        names = {0: 'Thumb', 1: 'Index', 2: 'Pinky'}
        return names.get(label, f'Unknown({label})')


@dataclass
class PairwiseCSPConfig:
    """Configuration for pairwise CSP feature extraction."""
    
    # Number of CSP components per pair (total features = n_components * n_pairs)
    # Typically use 4-6 components (takes 2-3 from each end of eigenvalue spectrum)
    n_components: int = 4
    
    # Regularization for CSP (helps with noisy/limited data)
    # Options: None, 'empirical', 'ledoit_wolf', 'oas', 'shrunk', 'pca', 'shrinkage'
    reg: Optional[str] = 'ledoit_wolf'
    
    # Whether to use log variance of CSP features (recommended)
    log: bool = True
    
    # Component selection strategy: 'mutual_info' or None
    # 'mutual_info' selects components based on mutual information with labels
    component_order: str = 'mutual_info'
    
    # Normalization: 'trace' (recommended), 'norm', or None
    norm_trace: bool = True
    
    # Class pairs for one-vs-one CSP
    # Each tuple is (class_a, class_b) where we fit CSP to separate class_a from class_b
    class_pairs: List[Tuple[int, int]] = field(default_factory=lambda: [
        (FingerLabels.THUMB, FingerLabels.INDEX),   # Thumb vs Index
        (FingerLabels.INDEX, FingerLabels.PINKY),   # Index vs Pinky
        (FingerLabels.THUMB, FingerLabels.PINKY),   # Thumb vs Pinky
    ])
    
    # EEG channel names (should match preprocessing output)
    eeg_channels: List[str] = field(default_factory=lambda: [
        'FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8'
    ])
    
    # Sampling rate (after preprocessing from clean.py)
    sample_rate: int = 125  # Hz
    
    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        return len(self.eeg_channels)
    
    @property
    def n_pairs(self) -> int:
        """Number of class pairs."""
        return len(self.class_pairs)
    
    @property
    def total_features(self) -> int:
        """Total number of CSP features (n_components * n_pairs)."""
        return self.n_components * self.n_pairs
    
    def get_pair_name(self, pair_idx: int) -> str:
        """Get human-readable name for a class pair."""
        if pair_idx >= len(self.class_pairs):
            return f'Pair{pair_idx}'
        class_a, class_b = self.class_pairs[pair_idx]
        return f'{FingerLabels.get_name(class_a)}_vs_{FingerLabels.get_name(class_b)}'


class PairwiseCSP(BaseEstimator, TransformerMixin):
    """
    Pairwise (One-vs-One) CSP for three-class motor imagery classification.
    
    Fits separate binary CSP models for each class pair:
    - Thumb vs Index
    - Index vs Pinky
    - Thumb vs Pinky
    
    Features from all pairs are concatenated for final classification.
    
    Parameters
    ----------
    config : PairwiseCSPConfig, optional
        Configuration object. Uses defaults if not provided.
    
    Examples
    --------
    Basic usage with windowed EEG data:
    
        >>> from control.csp_features import PairwiseCSP, PairwiseCSPConfig
        >>> 
        >>> # X shape: (n_windows, n_channels, n_samples) = (100, 8, 94)
        >>> # y shape: (n_windows,) with labels in {0, 1, 2}
        >>> X_train = np.random.randn(100, 8, 94)
        >>> y_train = np.random.randint(0, 3, 100)
        >>> 
        >>> csp = PairwiseCSP()
        >>> csp.fit(X_train, y_train)
        >>> 
        >>> # Transform to CSP features
        >>> X_features = csp.transform(X_train)
        >>> print(f"Feature shape: {X_features.shape}")  # (100, 12) for 4 components * 3 pairs
    
    Integration with sklearn Pipeline:
    
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.svm import SVC
        >>> 
        >>> clf = Pipeline([
        ...     ('csp', PairwiseCSP()),
        ...     ('svc', SVC(kernel='linear'))
        ... ])
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
    
    Attributes
    ----------
    csp_models_ : dict
        Dictionary of fitted CSP models, keyed by (class_a, class_b) tuples.
    pair_names_ : list
        Human-readable names for each class pair.
    is_fitted_ : bool
        Whether the model has been fitted.
    """
    
    def __init__(self, config: Optional[PairwiseCSPConfig] = None):
        """
        Initialize pairwise CSP extractor.
        
        Parameters
        ----------
        config : PairwiseCSPConfig, optional
            Configuration object. Uses defaults if not provided.
        """
        self.config = config or PairwiseCSPConfig()
        self._lock = threading.Lock()
        
        # Will be populated during fit()
        self.csp_models_: Dict[Tuple[int, int], CSP] = {}
        self.pair_names_: List[str] = []
        self.is_fitted_: bool = False
    
    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data shape and labels.
        
        Parameters
        ----------
        X : ndarray
            EEG data with shape (n_epochs, n_channels, n_samples).
        y : ndarray, optional
            Labels with shape (n_epochs,).
            
        Raises
        ------
        ValueError
            If input shape is invalid or labels are missing required classes.
        """
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3D (n_epochs, n_channels, n_samples), got shape {X.shape}"
            )
        
        n_epochs, n_channels, n_samples = X.shape
        
        if n_channels != self.config.n_channels:
            raise ValueError(
                f"Expected {self.config.n_channels} channels, got {n_channels}"
            )
        
        if y is not None:
            if len(y) != n_epochs:
                raise ValueError(
                    f"Number of labels ({len(y)}) doesn't match number of epochs ({n_epochs})"
                )
            
            # Check that all required classes are present
            unique_labels = set(np.unique(y))
            required_labels = set()
            for class_a, class_b in self.config.class_pairs:
                required_labels.add(class_a)
                required_labels.add(class_b)
            
            missing = required_labels - unique_labels
            if missing:
                raise ValueError(
                    f"Missing required class labels: {missing}. "
                    f"Found labels: {unique_labels}"
                )
    
    def _create_csp(self) -> CSP:
        """
        Create a new CSP instance with configured parameters.
        
        Returns
        -------
        CSP
            Configured MNE CSP object.
        """
        # MNE CSP API: log=True requires transform_into='average_power'
        # When transform_into='csp_space', log must be None
        if self.config.log:
            return CSP(
                n_components=self.config.n_components,
                reg=self.config.reg,
                log=True,
                norm_trace=self.config.norm_trace,
                component_order=self.config.component_order,
                transform_into='average_power'
            )
        else:
            return CSP(
                n_components=self.config.n_components,
                reg=self.config.reg,
                log=None,
                norm_trace=self.config.norm_trace,
                component_order=self.config.component_order,
                transform_into='csp_space'
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PairwiseCSP':
        """
        Fit pairwise CSP models to training data.
        
        Fits separate binary CSP for each class pair:
        - Thumb (0) vs Index (1)
        - Index (1) vs Pinky (2)
        - Thumb (0) vs Pinky (2)
        
        Parameters
        ----------
        X : ndarray
            Training EEG data with shape (n_epochs, n_channels, n_samples).
            Expected shape: (n_windows, 8, 94) for preprocessed Unicorn data.
        y : ndarray
            Training labels with shape (n_epochs,).
            Labels should be in {0: Thumb, 1: Index, 2: Pinky}.
            
        Returns
        -------
        self
            Fitted PairwiseCSP instance.
        """
        with self._lock:
            self._validate_input(X, y)
            
            self.csp_models_ = {}
            self.pair_names_ = []
            
            for pair_idx, (class_a, class_b) in enumerate(self.config.class_pairs):
                # Select data for this binary classification
                mask = np.isin(y, [class_a, class_b])
                X_pair = X[mask]
                y_pair = y[mask]
                
                # Create binary labels for CSP (CSP needs labels to be different)
                # Map to 0/1 for this pair
                y_binary = (y_pair == class_b).astype(int)
                
                # Fit CSP for this pair
                csp = self._create_csp()
                csp.fit(X_pair, y_binary)
                
                pair_key = (class_a, class_b)
                self.csp_models_[pair_key] = csp
                self.pair_names_.append(self.config.get_pair_name(pair_idx))
                
                n_samples_a = np.sum(y_pair == class_a)
                n_samples_b = np.sum(y_pair == class_b)
                print(f"Fitted CSP for {self.pair_names_[-1]}: "
                      f"{n_samples_a} vs {n_samples_b} samples")
            
            self.is_fitted_ = True
            return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform EEG data to pairwise CSP features.
        
        Applies all fitted CSP models and concatenates features.
        
        Parameters
        ----------
        X : ndarray
            EEG data with shape (n_epochs, n_channels, n_samples).
            
        Returns
        -------
        X_csp : ndarray
            CSP features with shape (n_epochs, n_components * n_pairs).
            For default config: (n_epochs, 12) = 4 components * 3 pairs.
        """
        if not self.is_fitted_:
            raise RuntimeError("PairwiseCSP must be fitted before transform()")
        
        self._validate_input(X)
        
        n_epochs = X.shape[0]
        features_list = []
        
        for pair_key in self.config.class_pairs:
            csp = self.csp_models_[pair_key]
            # Transform returns (n_epochs, n_components)
            pair_features = csp.transform(X)
            features_list.append(pair_features)
        
        # Concatenate features from all pairs
        # Shape: (n_epochs, n_components * n_pairs)
        X_csp = np.hstack(features_list)
        
        return X_csp
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit CSP models and transform data in one step.
        
        Parameters
        ----------
        X : ndarray
            Training EEG data with shape (n_epochs, n_channels, n_samples).
        y : ndarray
            Training labels with shape (n_epochs,).
            
        Returns
        -------
        X_csp : ndarray
            CSP features with shape (n_epochs, n_components * n_pairs).
        """
        return self.fit(X, y).transform(X)
    
    def get_spatial_patterns(self) -> Dict[str, np.ndarray]:
        """
        Get spatial patterns (activation patterns) for each class pair.
        
        Spatial patterns show which channels contribute most to each CSP component.
        These are useful for visualization and interpretation.
        
        Returns
        -------
        patterns : dict
            Dictionary mapping pair names to spatial pattern arrays.
            Each array has shape (n_components, n_channels).
        """
        if not self.is_fitted_:
            raise RuntimeError("PairwiseCSP must be fitted before getting patterns")
        
        patterns = {}
        for pair_idx, pair_key in enumerate(self.config.class_pairs):
            csp = self.csp_models_[pair_key]
            pair_name = self.pair_names_[pair_idx]
            patterns[pair_name] = csp.patterns_.T  # Transpose to (n_components, n_channels)
        
        return patterns
    
    def get_spatial_filters(self) -> Dict[str, np.ndarray]:
        """
        Get spatial filters for each class pair.
        
        Spatial filters are applied to the data during transformation.
        
        Returns
        -------
        filters : dict
            Dictionary mapping pair names to spatial filter arrays.
            Each array has shape (n_components, n_channels).
        """
        if not self.is_fitted_:
            raise RuntimeError("PairwiseCSP must be fitted before getting filters")
        
        filters = {}
        for pair_idx, pair_key in enumerate(self.config.class_pairs):
            csp = self.csp_models_[pair_key]
            pair_name = self.pair_names_[pair_idx]
            filters[pair_name] = csp.filters_[:self.config.n_components]
        
        return filters
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for CSP features.
        
        Returns
        -------
        names : list
            List of feature names, one per CSP feature.
        """
        names = []
        for pair_idx, pair_key in enumerate(self.config.class_pairs):
            pair_name = self.pair_names_[pair_idx] if self.is_fitted_ else self.config.get_pair_name(pair_idx)
            for comp in range(self.config.n_components):
                names.append(f'{pair_name}_csp{comp}')
        return names


class RealTimeCSPExtractor:
    """
    Real-time CSP feature extraction for streaming EEG data.
    
    Uses pre-fitted PairwiseCSP to extract features from sliding windows
    as they arrive during real-time classification.
    
    Parameters
    ----------
    csp_model : PairwiseCSP
        Pre-fitted pairwise CSP model.
    
    Examples
    --------
    Integration with windowing pipeline:
    
        >>> from unicorneeg.pipe import RealTimeWindowBuffer
        >>> from control.csp_features import PairwiseCSP, RealTimeCSPExtractor
        >>> 
        >>> # Train CSP model offline
        >>> csp = PairwiseCSP()
        >>> csp.fit(X_train, y_train)
        >>> 
        >>> # Create real-time extractor
        >>> extractor = RealTimeCSPExtractor(csp)
        >>> 
        >>> def handle_window(window):
        ...     features = extractor.extract(window.data)
        ...     # Feed features to classifier
        ...     prediction = classifier.predict(features.reshape(1, -1))
        >>> 
        >>> window_buffer = RealTimeWindowBuffer()
        >>> window_buffer.on_window(handle_window)
    """
    
    def __init__(self, csp_model: PairwiseCSP):
        """
        Initialize real-time CSP extractor.
        
        Parameters
        ----------
        csp_model : PairwiseCSP
            Pre-fitted pairwise CSP model.
        """
        if not csp_model.is_fitted_:
            raise ValueError("CSP model must be fitted before creating extractor")
        
        self.csp_model = csp_model
        self._lock = threading.Lock()
    
    def extract(self, window_data: np.ndarray) -> np.ndarray:
        """
        Extract CSP features from a single window.
        
        Parameters
        ----------
        window_data : ndarray
            Single window of EEG data.
            Shape: (n_samples, n_channels) = (94, 8) from windowing pipeline
            OR (n_channels, n_samples) = (8, 94) if already transposed.
            
        Returns
        -------
        features : ndarray
            CSP features with shape (n_components * n_pairs,) = (12,).
        """
        with self._lock:
            # Handle different input shapes
            if window_data.ndim == 2:
                n_dim0, n_dim1 = window_data.shape
                n_channels = self.csp_model.config.n_channels
                
                # Determine orientation and reshape to (1, n_channels, n_samples)
                if n_dim0 == n_channels:
                    # Already (n_channels, n_samples)
                    X = window_data[np.newaxis, :, :]
                elif n_dim1 == n_channels:
                    # (n_samples, n_channels) - need to transpose
                    X = window_data.T[np.newaxis, :, :]
                else:
                    raise ValueError(
                        f"Window shape {window_data.shape} doesn't match expected channels ({n_channels})"
                    )
            else:
                raise ValueError(
                    f"Window must be 2D, got shape {window_data.shape}"
                )
            
            # Transform returns (1, n_features), squeeze to (n_features,)
            features = self.csp_model.transform(X).squeeze()
            return features
    
    def extract_batch(self, windows: List[np.ndarray]) -> np.ndarray:
        """
        Extract CSP features from multiple windows.
        
        Parameters
        ----------
        windows : list of ndarray
            List of window data arrays.
            
        Returns
        -------
        features : ndarray
            CSP features with shape (n_windows, n_components * n_pairs).
        """
        features_list = [self.extract(w) for w in windows]
        return np.vstack(features_list)


def prepare_data_for_csp(
    windows: List,
    transpose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare windowed data for CSP training.
    
    Converts list of LabeledWindow objects (from pipe.py) to arrays
    suitable for CSP fitting.
    
    Parameters
    ----------
    windows : list
        List of LabeledWindow objects from SlidingWindowGenerator or
        RealTimeWindowBuffer.
    transpose : bool, optional
        Whether to transpose data from (n_samples, n_channels) to
        (n_channels, n_samples). Default True (CSP expects latter).
        
    Returns
    -------
    X : ndarray
        EEG data with shape (n_windows, n_channels, n_samples).
    y : ndarray
        Labels with shape (n_windows,).
    
    Examples
    --------
        >>> from unicorneeg.pipe import SlidingWindowGenerator
        >>> from control.csp_features import prepare_data_for_csp, PairwiseCSP
        >>> 
        >>> # Generate windows from trials
        >>> generator = SlidingWindowGenerator()
        >>> windows = generator.process_trials(trials)
        >>> 
        >>> # Prepare for CSP
        >>> X, y = prepare_data_for_csp(windows)
        >>> print(f"X shape: {X.shape}")  # (n_windows, 8, 94)
        >>> 
        >>> # Train CSP
        >>> csp = PairwiseCSP()
        >>> csp.fit(X, y)
    """
    data_list = []
    labels = []
    
    for window in windows:
        # Handle both LabeledWindow objects and dicts
        if hasattr(window, 'data'):
            data = window.data
            label = window.label
        elif isinstance(window, dict):
            data = window['data']
            label = window['label']
        else:
            raise ValueError(f"Unknown window type: {type(window)}")
        
        if transpose and data.shape[0] > data.shape[1]:
            # Data is (n_samples, n_channels), transpose to (n_channels, n_samples)
            data = data.T
        
        data_list.append(data)
        labels.append(label)
    
    X = np.array(data_list)
    y = np.array(labels)
    
    return X, y


def create_csp_pipeline(
    csp_config: Optional[PairwiseCSPConfig] = None,
    classifier: Optional[BaseEstimator] = None
) -> Pipeline:
    """
    Create a sklearn Pipeline with pairwise CSP and classifier.
    
    Parameters
    ----------
    csp_config : PairwiseCSPConfig, optional
        Configuration for CSP. Uses defaults if not provided.
    classifier : BaseEstimator, optional
        Sklearn classifier. Defaults to LinearSVC if not provided.
        
    Returns
    -------
    pipeline : Pipeline
        Sklearn pipeline with CSP feature extraction and classification.
    
    Examples
    --------
        >>> from control.csp_features import create_csp_pipeline
        >>> from sklearn.model_selection import cross_val_score
        >>> 
        >>> pipeline = create_csp_pipeline()
        >>> scores = cross_val_score(pipeline, X, y, cv=5)
        >>> print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
    """
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    
    if classifier is None:
        classifier = LinearSVC(dual='auto')
    
    csp = PairwiseCSP(config=csp_config)
    
    pipeline = Pipeline([
        ('csp', csp),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    
    return pipeline


# Testing and demonstration
if __name__ == '__main__':
    print("=" * 60)
    print("Pairwise CSP Feature Extraction - Demo")
    print("=" * 60)
    
    # Create synthetic data for testing
    np.random.seed(42)
    
    n_epochs_per_class = 50
    n_channels = 8
    n_samples = 94  # 750ms at 125Hz
    
    # Simulate data with different spatial patterns for each class
    # In real data, these differences would come from motor imagery
    X_list = []
    y_list = []
    
    # Simulated spatial patterns (which channels are active for each class)
    # Thumb: C3 dominant (channel 1)
    # Index: CZ dominant (channel 2)
    # Pinky: C4 dominant (channel 3)
    
    for label in [FingerLabels.THUMB, FingerLabels.INDEX, FingerLabels.PINKY]:
        for _ in range(n_epochs_per_class):
            # Base noise
            epoch = np.random.randn(n_channels, n_samples) * 0.5
            
            # Add class-specific spatial pattern
            if label == FingerLabels.THUMB:
                epoch[1, :] += np.sin(2 * np.pi * 10 * np.arange(n_samples) / 125) * 2
            elif label == FingerLabels.INDEX:
                epoch[2, :] += np.sin(2 * np.pi * 12 * np.arange(n_samples) / 125) * 2
            elif label == FingerLabels.PINKY:
                epoch[3, :] += np.sin(2 * np.pi * 11 * np.arange(n_samples) / 125) * 2
            
            X_list.append(epoch)
            y_list.append(label)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\nSynthetic data shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)} (Thumb=0, Index=1, Pinky=2)")
    
    # Fit pairwise CSP
    print("\n" + "-" * 40)
    print("Fitting Pairwise CSP...")
    print("-" * 40)
    
    config = PairwiseCSPConfig(n_components=4)
    csp = PairwiseCSP(config=config)
    X_csp = csp.fit_transform(X, y)
    
    print(f"\nCSP feature shape: {X_csp.shape}")
    print(f"Feature names: {csp.get_feature_names()[:6]}...")
    
    # Get spatial patterns
    patterns = csp.get_spatial_patterns()
    print(f"\nSpatial patterns available for: {list(patterns.keys())}")
    
    # Test real-time extraction
    print("\n" + "-" * 40)
    print("Testing Real-Time Extraction...")
    print("-" * 40)
    
    extractor = RealTimeCSPExtractor(csp)
    
    # Simulate a single window (as would come from pipe.py)
    test_window = X[0].T  # (n_samples, n_channels) format from windowing
    features = extractor.extract(test_window)
    print(f"Single window features shape: {features.shape}")
    print(f"Features: {features[:4]}...")  # First 4 features
    
    # Test classification pipeline
    print("\n" + "-" * 40)
    print("Testing Classification Pipeline...")
    print("-" * 40)
    
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    pipeline = create_csp_pipeline(config)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    
    print(f"5-fold CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"Individual folds: {scores}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
