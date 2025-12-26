"""
Pairwise Common Spatial Pattern (CSP) Feature Extraction Module.

Implements one-vs-one pairwise CSP for three-class motor imagery classification:
- Thumb vs Index
- Index vs Pinky  
- Thumb vs Pinky

Features are extracted separately for alpha (8-13 Hz) and beta (13-30 Hz) bands,
producing 24 total features: 4 components × 3 pairs × 2 bands.

CSP finds spatial filters that maximize variance for one class while
minimizing variance for another class. For three classes, we use
pairwise binary CSP rather than multiclass CSP.

Pipeline:
1. Apply CSP spatial filters to windowed EEG data (per band)
2. Compute variance over window for each CSP-projected signal
3. Apply log transform to variance features
4. Apply session-wise z-scoring (offline: compute mean/std, online: use stored)

Uses MNE-Python for CSP implementation.

References:
    Blankertz et al., "Optimizing Spatial filters for Robust EEG 
    Single-Trial Analysis", IEEE Signal Processing Magazine, 2008.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Callable
import threading
import pickle
from pathlib import Path

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


@dataclass
class DualBandCSPConfig:
    """Configuration for dual-band pairwise CSP feature extraction.
    
    Total features = n_components × n_pairs × n_bands = 4 × 3 × 2 = 24
    """
    
    # Number of CSP components per pair per band
    n_components: int = 4
    
    # Regularization for CSP
    reg: Optional[str] = 'ledoit_wolf'
    
    # Component selection strategy
    component_order: str = 'mutual_info'
    
    # Normalization
    norm_trace: bool = True
    
    # Class pairs for one-vs-one CSP
    class_pairs: List[Tuple[int, int]] = field(default_factory=lambda: [
        (FingerLabels.THUMB, FingerLabels.INDEX),
        (FingerLabels.INDEX, FingerLabels.PINKY),
        (FingerLabels.THUMB, FingerLabels.PINKY),
    ])
    
    # EEG channel names (should match preprocessing output from clean.py)
    eeg_channels: List[str] = field(default_factory=lambda: [
        'FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8'
    ])
    
    # Frequency bands
    bands: List[str] = field(default_factory=lambda: ['alpha', 'beta'])
    
    # Sampling rate (after preprocessing from clean.py)
    sample_rate: int = 125
    
    # Epsilon for log transform to avoid log(0)
    log_epsilon: float = 1e-10
    
    # Z-score epsilon for numerical stability
    zscore_epsilon: float = 1e-8
    
    @property
    def n_channels(self) -> int:
        return len(self.eeg_channels)
    
    @property
    def n_pairs(self) -> int:
        return len(self.class_pairs)
    
    @property
    def n_bands(self) -> int:
        return len(self.bands)
    
    @property
    def total_features(self) -> int:
        """Total features: n_components × n_pairs × n_bands = 24."""
        return self.n_components * self.n_pairs * self.n_bands
    
    @property
    def features_per_band(self) -> int:
        """Features per band: n_components × n_pairs = 12."""
        return self.n_components * self.n_pairs
    
    def get_pair_name(self, pair_idx: int) -> str:
        if pair_idx >= len(self.class_pairs):
            return f'Pair{pair_idx}'
        class_a, class_b = self.class_pairs[pair_idx]
        return f'{FingerLabels.get_name(class_a)}_vs_{FingerLabels.get_name(class_b)}'
    
    def get_alpha_channels(self) -> List[str]:
        """Get alpha band channel names (as output by clean.py)."""
        return [f'alpha_{ch}' for ch in self.eeg_channels]
    
    def get_beta_channels(self) -> List[str]:
        """Get beta band channel names (as output by clean.py)."""
        return [f'beta_{ch}' for ch in self.eeg_channels]


class SessionNormalizer:
    """
    Session-wise z-score normalization for CSP features.
    
    Computes mean and std during offline training, stores them
    for consistent normalization during online inference.
    """
    
    def __init__(self, n_features: int = 24, epsilon: float = 1e-8):
        self.n_features = n_features
        self.epsilon = epsilon
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False
    
    def fit(self, X: np.ndarray) -> 'SessionNormalizer':
        """Compute mean and std from training data."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where(self.std_ < self.epsilon, self.epsilon, self.std_)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization."""
        if not self.is_fitted_:
            raise RuntimeError("Normalizer must be fitted before transform()")
        
        squeeze = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            squeeze = True
        
        X_normalized = (X - self.mean_) / self.std_
        
        if squeeze:
            X_normalized = X_normalized.squeeze()
        return X_normalized
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse z-score normalization."""
        if not self.is_fitted_:
            raise RuntimeError("Normalizer must be fitted before inverse_transform()")
        
        squeeze = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            squeeze = True
        
        X_original = X * self.std_ + self.mean_
        
        if squeeze:
            X_original = X_original.squeeze()
        return X_original
    
    def save(self, path: Union[str, Path]) -> None:
        """Save normalizer statistics to file."""
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump({
                'n_features': self.n_features,
                'epsilon': self.epsilon,
                'mean': self.mean_,
                'std': self.std_,
                'is_fitted': self.is_fitted_
            }, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SessionNormalizer':
        """Load normalizer from file."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        normalizer = cls(n_features=data['n_features'], epsilon=data['epsilon'])
        normalizer.mean_ = data['mean']
        normalizer.std_ = data['std']
        normalizer.is_fitted_ = data['is_fitted']
        return normalizer


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


class DualBandPairwiseCSP(BaseEstimator, TransformerMixin):
    """
    Dual-band Pairwise CSP with log-variance features and session z-scoring.
    
    Fits separate binary CSP models for each class pair AND each frequency band:
    - Alpha band (8-13 Hz): Thumb vs Index, Index vs Pinky, Thumb vs Pinky
    - Beta band (13-30 Hz): Thumb vs Index, Index vs Pinky, Thumb vs Pinky
    
    For each CSP-filtered signal, computes:
    1. Spatial filtering via CSP
    2. Variance over the window
    3. Log transform of variance
    4. Session-wise z-scoring (optional)
    
    Total features: 4 components × 3 pairs × 2 bands = 24
    
    Parameters
    ----------
    config : DualBandCSPConfig, optional
        Configuration object. Uses defaults if not provided.
    enable_zscore : bool
        Whether to apply z-score normalization. Default True.
    """
    
    def __init__(
        self,
        config: Optional[DualBandCSPConfig] = None,
        enable_zscore: bool = True
    ):
        self.config = config or DualBandCSPConfig()
        self.enable_zscore = enable_zscore
        self._lock = threading.Lock()
        
        # Populated during fit()
        self.alpha_csp_models_: Dict[Tuple[int, int], CSP] = {}
        self.beta_csp_models_: Dict[Tuple[int, int], CSP] = {}
        self.normalizer_: Optional[SessionNormalizer] = None
        self.pair_names_: List[str] = []
        self.is_fitted_: bool = False
    
    def _validate_input(
        self,
        X_alpha: np.ndarray,
        X_beta: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> None:
        """Validate input data shape and labels."""
        for name, X in [('X_alpha', X_alpha), ('X_beta', X_beta)]:
            if X.ndim != 3:
                raise ValueError(
                    f"{name} must be 3D (n_epochs, n_channels, n_samples), got shape {X.shape}"
                )
            if X.shape[1] != self.config.n_channels:
                raise ValueError(
                    f"{name}: Expected {self.config.n_channels} channels, got {X.shape[1]}"
                )
        
        if X_alpha.shape != X_beta.shape:
            raise ValueError(
                f"X_alpha and X_beta must have same shape. "
                f"Got {X_alpha.shape} and {X_beta.shape}"
            )
        
        if y is not None:
            n_epochs = X_alpha.shape[0]
            if len(y) != n_epochs:
                raise ValueError(
                    f"Number of labels ({len(y)}) doesn't match epochs ({n_epochs})"
                )
            
            unique_labels = set(np.unique(y))
            required_labels = set()
            for class_a, class_b in self.config.class_pairs:
                required_labels.add(class_a)
                required_labels.add(class_b)
            
            missing = required_labels - unique_labels
            if missing:
                raise ValueError(f"Missing required class labels: {missing}")
    
    def _create_csp(self) -> CSP:
        """Create CSP instance that returns raw filtered signals."""
        return CSP(
            n_components=self.config.n_components,
            reg=self.config.reg,
            log=None,
            norm_trace=self.config.norm_trace,
            component_order=self.config.component_order,
            transform_into='csp_space'
        )
    
    def _compute_log_variance(self, X_csp: np.ndarray) -> np.ndarray:
        """
        Compute log variance of CSP-filtered signals.
        
        Parameters
        ----------
        X_csp : ndarray
            CSP-filtered data (n_epochs, n_components, n_samples).
            
        Returns
        -------
        log_var : ndarray
            Log variance features (n_epochs, n_components).
        """
        variance = np.var(X_csp, axis=-1)
        log_var = np.log(variance + self.config.log_epsilon)
        return log_var
    
    def _fit_band(
        self,
        X: np.ndarray,
        y: np.ndarray,
        band_name: str
    ) -> Dict[Tuple[int, int], CSP]:
        """Fit CSP models for one frequency band."""
        csp_models = {}
        
        for pair_idx, (class_a, class_b) in enumerate(self.config.class_pairs):
            mask = np.isin(y, [class_a, class_b])
            X_pair = X[mask]
            y_pair = y[mask]
            y_binary = (y_pair == class_b).astype(int)
            
            csp = self._create_csp()
            csp.fit(X_pair, y_binary)
            
            pair_key = (class_a, class_b)
            csp_models[pair_key] = csp
            
            n_a = np.sum(y_pair == class_a)
            n_b = np.sum(y_pair == class_b)
            pair_name = self.config.get_pair_name(pair_idx)
            print(f"[{band_name.upper()}] Fitted CSP for {pair_name}: {n_a} vs {n_b} samples")
        
        return csp_models
    
    def _transform_band(
        self,
        X: np.ndarray,
        csp_models: Dict[Tuple[int, int], CSP]
    ) -> np.ndarray:
        """Transform data through CSP and compute log-variance for one band."""
        features_list = []
        
        for pair_key in self.config.class_pairs:
            csp = csp_models[pair_key]
            X_csp = csp.transform(X)  # (n_epochs, n_components, n_samples)
            log_var = self._compute_log_variance(X_csp)  # (n_epochs, n_components)
            features_list.append(log_var)
        
        return np.hstack(features_list)  # (n_epochs, n_components * n_pairs)
    
    def fit(
        self,
        X_alpha: np.ndarray,
        X_beta: np.ndarray,
        y: np.ndarray
    ) -> 'DualBandPairwiseCSP':
        """
        Fit dual-band CSP models to training data.
        
        Parameters
        ----------
        X_alpha : ndarray
            Alpha band EEG data (n_epochs, n_channels, n_samples).
        X_beta : ndarray
            Beta band EEG data (n_epochs, n_channels, n_samples).
        y : ndarray
            Labels (n_epochs,) with values in {0: Thumb, 1: Index, 2: Pinky}.
        """
        with self._lock:
            self._validate_input(X_alpha, X_beta, y)
            
            print(f"\n{'='*50}")
            print("Fitting Dual-Band Pairwise CSP")
            print(f"{'='*50}")
            print(f"Input shape: {X_alpha.shape} per band")
            
            print(f"\n{'-'*30}")
            self.alpha_csp_models_ = self._fit_band(X_alpha, y, 'alpha')
            
            print(f"\n{'-'*30}")
            self.beta_csp_models_ = self._fit_band(X_beta, y, 'beta')
            
            self.pair_names_ = [
                self.config.get_pair_name(i) 
                for i in range(len(self.config.class_pairs))
            ]
            
            # Fit z-score normalizer
            if self.enable_zscore:
                print(f"\n{'-'*30}")
                print("Fitting session normalizer...")
                
                alpha_features = self._transform_band(X_alpha, self.alpha_csp_models_)
                beta_features = self._transform_band(X_beta, self.beta_csp_models_)
                all_features = np.hstack([alpha_features, beta_features])
                
                self.normalizer_ = SessionNormalizer(
                    n_features=self.config.total_features,
                    epsilon=self.config.zscore_epsilon
                )
                self.normalizer_.fit(all_features)
                
                print(f"Normalizer fitted on {all_features.shape[0]} samples, "
                      f"{all_features.shape[1]} features")
            
            self.is_fitted_ = True
            
            print(f"\n{'='*50}")
            print(f"Total features: {self.config.total_features}")
            print(f"  Alpha band: {self.config.features_per_band} features")
            print(f"  Beta band: {self.config.features_per_band} features")
            print(f"{'='*50}")
            
            return self
    
    def transform(
        self,
        X_alpha: np.ndarray,
        X_beta: np.ndarray,
        apply_zscore: Optional[bool] = None
    ) -> np.ndarray:
        """
        Transform EEG data to 24 dual-band CSP features.
        
        Pipeline per band:
        1. Apply CSP spatial filters
        2. Compute variance over window
        3. Apply log transform
        4. Apply z-score normalization (if enabled)
        
        Parameters
        ----------
        X_alpha : ndarray
            Alpha band data (n_epochs, n_channels, n_samples).
        X_beta : ndarray
            Beta band data (n_epochs, n_channels, n_samples).
        apply_zscore : bool, optional
            Override z-scoring. If None, uses self.enable_zscore.
            
        Returns
        -------
        features : ndarray
            CSP features (n_epochs, 24). First 12 = alpha, last 12 = beta.
        """
        if not self.is_fitted_:
            raise RuntimeError("Must be fitted before transform()")
        
        self._validate_input(X_alpha, X_beta)
        
        alpha_features = self._transform_band(X_alpha, self.alpha_csp_models_)
        beta_features = self._transform_band(X_beta, self.beta_csp_models_)
        features = np.hstack([alpha_features, beta_features])
        
        should_zscore = apply_zscore if apply_zscore is not None else self.enable_zscore
        if should_zscore and self.normalizer_ is not None:
            features = self.normalizer_.transform(features)
        
        return features
    
    def fit_transform(
        self,
        X_alpha: np.ndarray,
        X_beta: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X_alpha, X_beta, y).transform(X_alpha, X_beta)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for all 24 CSP features."""
        names = []
        for band in self.config.bands:
            for pair_idx in range(self.config.n_pairs):
                pair_name = self.config.get_pair_name(pair_idx)
                for comp in range(self.config.n_components):
                    names.append(f'{band}_{pair_name}_csp{comp}')
        return names
    
    def get_spatial_patterns(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get spatial patterns for visualization."""
        if not self.is_fitted_:
            raise RuntimeError("Must be fitted before getting patterns")
        
        patterns = {'alpha': {}, 'beta': {}}
        for pair_idx, pair_key in enumerate(self.config.class_pairs):
            pair_name = self.pair_names_[pair_idx]
            patterns['alpha'][pair_name] = self.alpha_csp_models_[pair_key].patterns_.T
            patterns['beta'][pair_name] = self.beta_csp_models_[pair_key].patterns_.T
        return patterns
    
    def save(self, path: Union[str, Path]) -> None:
        """Save fitted model to file."""
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'enable_zscore': self.enable_zscore,
                'alpha_csp_models': self.alpha_csp_models_,
                'beta_csp_models': self.beta_csp_models_,
                'normalizer': self.normalizer_,
                'pair_names': self.pair_names_,
                'is_fitted': self.is_fitted_
            }, f)
        print(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DualBandPairwiseCSP':
        """Load fitted model from file."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(config=data['config'], enable_zscore=data['enable_zscore'])
        model.alpha_csp_models_ = data['alpha_csp_models']
        model.beta_csp_models_ = data['beta_csp_models']
        model.normalizer_ = data['normalizer']
        model.pair_names_ = data['pair_names']
        model.is_fitted_ = data['is_fitted']
        
        print(f"Loaded model from {path}")
        return model


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


class DualBandRealTimeCSPExtractor:
    """
    Real-time dual-band CSP feature extraction for streaming EEG data.
    
    Uses pre-fitted DualBandPairwiseCSP to extract 24 features from
    sliding windows as they arrive during real-time classification.
    
    Parameters
    ----------
    csp_model : DualBandPairwiseCSP
        Pre-fitted dual-band CSP model.
    
    Examples
    --------
    Integration with multiband windowing pipeline:
    
        >>> from unicorneeg.pipe import RealTimeWindowBuffer, WindowConfig
        >>> from control.csp_features import DualBandPairwiseCSP, DualBandRealTimeCSPExtractor
        >>> 
        >>> # Train CSP model offline
        >>> csp = DualBandPairwiseCSP()
        >>> csp.fit(X_alpha_train, X_beta_train, y_train)
        >>> 
        >>> # Create real-time extractor
        >>> extractor = DualBandRealTimeCSPExtractor(csp)
        >>> 
        >>> # Setup multiband windowing
        >>> config = WindowConfig(use_multiband=True)
        >>> window_buffer = RealTimeWindowBuffer(config=config)
        >>> 
        >>> def handle_window(window):
        ...     # window.data shape: (94, 16) for multiband
        ...     features = extractor.extract_from_multiband_window(window.data)
        ...     # features shape: (24,)
        ...     prediction = classifier.predict(features.reshape(1, -1))
        >>> 
        >>> window_buffer.on_window(handle_window)
    """
    
    def __init__(self, csp_model: DualBandPairwiseCSP):
        """Initialize real-time dual-band CSP extractor."""
        if not csp_model.is_fitted_:
            raise ValueError("CSP model must be fitted before creating extractor")
        
        self.csp_model = csp_model
        self._lock = threading.Lock()
    
    def _prepare_window(self, window: np.ndarray, n_channels: int) -> np.ndarray:
        """Prepare window data for CSP transform."""
        if window.ndim != 2:
            raise ValueError(f"Window must be 2D, got shape {window.shape}")
        
        n_dim0, n_dim1 = window.shape
        
        if n_dim0 == n_channels:
            return window[np.newaxis, :, :]
        elif n_dim1 == n_channels:
            return window.T[np.newaxis, :, :]
        else:
            raise ValueError(
                f"Window shape {window.shape} doesn't match expected channels ({n_channels})"
            )
    
    def extract(
        self,
        window_alpha: np.ndarray,
        window_beta: np.ndarray
    ) -> np.ndarray:
        """
        Extract 24 CSP features from separate alpha and beta windows.
        
        Parameters
        ----------
        window_alpha : ndarray
            Alpha band window data (n_samples, n_channels) = (94, 8).
        window_beta : ndarray
            Beta band window data (n_samples, n_channels) = (94, 8).
            
        Returns
        -------
        features : ndarray
            24 CSP features with shape (24,).
        """
        with self._lock:
            n_channels = self.csp_model.config.n_channels
            
            X_alpha = self._prepare_window(window_alpha, n_channels)
            X_beta = self._prepare_window(window_beta, n_channels)
            
            features = self.csp_model.transform(X_alpha, X_beta).squeeze()
            return features
    
    def extract_from_multiband_window(self, window_data: np.ndarray) -> np.ndarray:
        """
        Extract 24 CSP features from a multiband window.
        
        The multiband window (from pipe.py with use_multiband=True) contains
        both alpha and beta channels:
        [alpha_FZ, ..., alpha_PO8, beta_FZ, ..., beta_PO8]
        
        Parameters
        ----------
        window_data : ndarray
            Multiband window data (n_samples, n_channels*2) = (94, 16).
            
        Returns
        -------
        features : ndarray
            24 CSP features with shape (24,).
        """
        n_channels = self.csp_model.config.n_channels
        
        # Split: first 8 channels = alpha, last 8 = beta
        window_alpha = window_data[:, :n_channels]  # (94, 8)
        window_beta = window_data[:, n_channels:]   # (94, 8)
        
        return self.extract(window_alpha, window_beta)
    
    def extract_batch(
        self,
        windows_alpha: List[np.ndarray],
        windows_beta: List[np.ndarray]
    ) -> np.ndarray:
        """Extract features from multiple window pairs."""
        features_list = [
            self.extract(wa, wb) 
            for wa, wb in zip(windows_alpha, windows_beta)
        ]
        return np.vstack(features_list)


def prepare_dual_band_data(
    windows: List,
    n_channels_per_band: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare windowed multiband data for DualBandPairwiseCSP training.
    
    Converts list of LabeledWindow objects (from pipe.py with use_multiband=True)
    to separate alpha and beta arrays suitable for fitting.
    
    Parameters
    ----------
    windows : list
        List of LabeledWindow objects from RealTimeWindowBuffer or
        SlidingWindowGenerator (with use_multiband=True).
    n_channels_per_band : int
        Number of channels per band (default 8 for Unicorn).
        
    Returns
    -------
    X_alpha : ndarray
        Alpha band data (n_windows, n_channels, n_samples).
    X_beta : ndarray
        Beta band data (n_windows, n_channels, n_samples).
    y : ndarray
        Labels (n_windows,).
    
    Examples
    --------
        >>> from unicorneeg.pipe import SlidingWindowGenerator, WindowConfig
        >>> from control.csp_features import prepare_dual_band_data, DualBandPairwiseCSP
        >>> 
        >>> # Setup multiband windowing
        >>> config = WindowConfig(use_multiband=True)
        >>> generator = SlidingWindowGenerator(config=config)
        >>> windows = generator.process_trials(trials)
        >>> 
        >>> # Prepare for dual-band CSP
        >>> X_alpha, X_beta, y = prepare_dual_band_data(windows)
        >>> 
        >>> # Train CSP
        >>> csp = DualBandPairwiseCSP()
        >>> csp.fit(X_alpha, X_beta, y)
    """
    alpha_list = []
    beta_list = []
    labels = []
    
    for window in windows:
        if hasattr(window, 'data'):
            data = window.data
            label = window.label
        elif isinstance(window, dict):
            data = window['data']
            label = window['label']
        else:
            raise ValueError(f"Unknown window type: {type(window)}")
        
        # data shape is (n_samples, n_channels*2) = (94, 16) for multiband
        if data.shape[1] == n_channels_per_band * 2:
            alpha_data = data[:, :n_channels_per_band]  # (94, 8)
            beta_data = data[:, n_channels_per_band:]   # (94, 8)
        elif data.shape[1] == n_channels_per_band:
            raise ValueError(
                "Single-band data detected. Use WindowConfig(use_multiband=True) "
                "for dual-band CSP."
            )
        else:
            raise ValueError(
                f"Unexpected channel count: {data.shape[1]}. "
                f"Expected {n_channels_per_band} or {n_channels_per_band * 2}."
            )
        
        # Transpose to (n_channels, n_samples) for CSP
        alpha_list.append(alpha_data.T)
        beta_list.append(beta_data.T)
        labels.append(label)
    
    X_alpha = np.array(alpha_list)  # (n_windows, 8, 94)
    X_beta = np.array(beta_list)    # (n_windows, 8, 94)
    y = np.array(labels)
    
    return X_alpha, X_beta, y


def create_csp_feature_callback(
    csp_model: DualBandPairwiseCSP,
    output_callback: Optional[Callable[[np.ndarray, int], None]] = None
) -> Callable:
    """
    Create a callback for real-time CSP feature extraction.
    
    Integrates with the windowing pipeline from pipe.py.
    
    Parameters
    ----------
    csp_model : DualBandPairwiseCSP
        Pre-fitted dual-band CSP model.
    output_callback : callable, optional
        Function to call with (features, label) after extraction.
        If None, features are returned directly.
        
    Returns
    -------
    callback : callable
        Callback function for RealTimeWindowBuffer.on_window().
    
    Examples
    --------
        >>> from unicorneeg.pipe import RealTimeWindowBuffer, WindowConfig
        >>> from control.csp_features import DualBandPairwiseCSP, create_csp_feature_callback
        >>> 
        >>> # Train CSP offline
        >>> csp = DualBandPairwiseCSP()
        >>> csp.fit(X_alpha_train, X_beta_train, y_train)
        >>> 
        >>> # Create feature callback
        >>> def on_features(features, label):
        ...     prediction = classifier.predict(features.reshape(1, -1))
        ...     print(f"Predicted: {prediction[0]}, Label: {label}")
        >>> 
        >>> feature_callback = create_csp_feature_callback(csp, on_features)
        >>> 
        >>> # Setup multiband windowing
        >>> config = WindowConfig(use_multiband=True)
        >>> buffer = RealTimeWindowBuffer(config=config)
        >>> buffer.on_window(feature_callback)
    """
    extractor = DualBandRealTimeCSPExtractor(csp_model)
    
    def callback(window):
        """Extract CSP features from window and pass to output callback."""
        features = extractor.extract_from_multiband_window(window.data)
        
        if output_callback is not None:
            output_callback(features, window.label)
        
        return features
    
    return callback


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
    print("=" * 70)
    print("Dual-Band Pairwise CSP Feature Extraction - Demo")
    print("=" * 70)
    
    # Create synthetic dual-band data for testing
    np.random.seed(42)
    
    n_epochs_per_class = 50
    n_channels = 8
    n_samples = 94  # 750ms at 125Hz
    
    # Simulate data with different spatial patterns for each class
    X_alpha_list = []
    X_beta_list = []
    y_list = []
    
    # Simulated spatial patterns:
    # Thumb: C3 dominant in alpha (channel 1)
    # Index: CZ dominant in beta (channel 2)
    # Pinky: C4 dominant in both (channel 3)
    
    for label in [FingerLabels.THUMB, FingerLabels.INDEX, FingerLabels.PINKY]:
        for _ in range(n_epochs_per_class):
            # Base noise for both bands
            alpha_epoch = np.random.randn(n_channels, n_samples) * 0.5
            beta_epoch = np.random.randn(n_channels, n_samples) * 0.5
            
            # Add class-specific patterns
            if label == FingerLabels.THUMB:
                alpha_epoch[1, :] += np.sin(2 * np.pi * 10 * np.arange(n_samples) / 125) * 2
            elif label == FingerLabels.INDEX:
                beta_epoch[2, :] += np.sin(2 * np.pi * 20 * np.arange(n_samples) / 125) * 2
            elif label == FingerLabels.PINKY:
                alpha_epoch[3, :] += np.sin(2 * np.pi * 11 * np.arange(n_samples) / 125) * 1.5
                beta_epoch[3, :] += np.sin(2 * np.pi * 18 * np.arange(n_samples) / 125) * 1.5
            
            X_alpha_list.append(alpha_epoch)
            X_beta_list.append(beta_epoch)
            y_list.append(label)
    
    X_alpha = np.array(X_alpha_list)
    X_beta = np.array(X_beta_list)
    y = np.array(y_list)
    
    print(f"\nSynthetic dual-band data shapes:")
    print(f"  X_alpha: {X_alpha.shape}")
    print(f"  X_beta: {X_beta.shape}")
    print(f"  y: {y.shape}")
    print(f"Classes: {np.unique(y)} (Thumb=0, Index=1, Pinky=2)")
    
    # Fit dual-band CSP
    config = DualBandCSPConfig(n_components=4)
    csp = DualBandPairwiseCSP(config=config, enable_zscore=True)
    
    features = csp.fit_transform(X_alpha, X_beta, y)
    
    print(f"\nFeature shape: {features.shape} (expected: 150 × 24)")
    print(f"Feature names (alpha): {csp.get_feature_names()[:4]}...")
    print(f"Feature names (beta): ...{csp.get_feature_names()[-4:]}")
    
    # Verify z-scoring
    print(f"\nZ-score verification:")
    print(f"  Feature mean (should be ~0): {np.mean(features, axis=0)[:3].round(3)}")
    print(f"  Feature std (should be ~1): {np.std(features, axis=0)[:3].round(3)}")
    
    # Test real-time extraction
    print(f"\n{'-'*50}")
    print("Testing Dual-Band Real-Time Extraction...")
    print(f"{'-'*50}")
    
    extractor = DualBandRealTimeCSPExtractor(csp)
    
    # Simulate a multiband window (as from pipe.py with use_multiband=True)
    test_alpha = X_alpha[0].T  # (94, 8)
    test_beta = X_beta[0].T    # (94, 8)
    test_multiband = np.hstack([test_alpha, test_beta])  # (94, 16)
    
    features_single = extractor.extract_from_multiband_window(test_multiband)
    print(f"Single window features shape: {features_single.shape}")
    print(f"First 4 alpha features: {features_single[:4].round(3)}")
    print(f"First 4 beta features: {features_single[12:16].round(3)}")
    
    # Test save/load
    print(f"\n{'-'*50}")
    print("Testing Save/Load...")
    print(f"{'-'*50}")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / 'csp_model.pkl'
        
        csp.save(model_path)
        csp_loaded = DualBandPairwiseCSP.load(model_path)
        
        features_loaded = csp_loaded.transform(X_alpha[:5], X_beta[:5])
        features_original = csp.transform(X_alpha[:5], X_beta[:5])
        
        assert np.allclose(features_loaded, features_original), "Save/load failed!"
        print("Save/load verification passed!")
    
    # Test classification
    print(f"\n{'-'*50}")
    print("Testing Classification with 24 Features...")
    print(f"{'-'*50}")
    
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.svm import LinearSVC
    
    clf = LinearSVC(dual='auto')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, features, y, cv=cv, scoring='accuracy')
    
    print(f"5-fold CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"Individual folds: {scores}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Dual-Band Pairwise CSP Features")
    print(f"{'='*70}")
    print(f"Total features: {csp.config.total_features}")
    print(f"  - Alpha band: {csp.config.features_per_band} features")
    print(f"    (Thumb vs Index, Index vs Pinky, Thumb vs Pinky × 4 components)")
    print(f"  - Beta band: {csp.config.features_per_band} features")
    print(f"    (Thumb vs Index, Index vs Pinky, Thumb vs Pinky × 4 components)")
    print(f"Session z-scoring: {'Enabled' if csp.enable_zscore else 'Disabled'}")
    print(f"{'='*70}")
    print("Demo Complete!")
    print(f"{'='*70}")
