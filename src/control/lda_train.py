"""
Training pipeline for CSP + LDA classifier from collected EEG data.

Trial structure:
    5s Rest → 1s Cue → 3s Motor Imagery [REPEAT for 320 trials]
    
The Motor Imagery period (3s) is used for training.

Input files:
    - CSV: Raw EEG data collected with 'collect' mode (~250Hz)
    - TXT: Cue order file with one cue per line (Thumb, Index, Pinky, or Rest)
           Note: "Rest" cues are parsed but SKIPPED during analysis/training

Output:
    - csp_model.pkl: Fitted DualBandPairwiseCSP
    - lda_classifier.pkl: Fitted CSPLDAClassifier
    - normalizer.pkl: SessionNormalizer for z-scoring
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unicorneeg.clean import EEGPreprocessor, PreprocessingConfig
from unicorneeg.pipe import SlidingWindowGenerator, WindowConfig, LabeledWindow
from control.csp_features import (
    DualBandPairwiseCSP,
    DualBandCSPConfig,
    SessionNormalizer,
    prepare_dual_band_data
)
from control.lda_classifier import CSPLDAClassifier


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrialConfig:
    """Configuration for trial structure."""
    
    # Trial timing (in seconds)
    rest_duration: float = 5.0       # Initial rest period
    cue_duration: float = 1.0        # Cue presentation
    mi_duration: float = 3.0         # Motor imagery (training data)
    
    # Sampling rate (raw data from Unicorn)
    raw_sample_rate: int = 250       # Hz (native Unicorn rate)
    
    # Sampling rate after preprocessing
    processed_sample_rate: int = 125  # Hz (after downsampling)
    
    # Number of trials
    n_trials: int = 320
    
    # Class labels
    class_labels: List[str] = None
    
    def __post_init__(self):
        if self.class_labels is None:
            self.class_labels = ['Thumb', 'Index', 'Pinky']
    
    @property
    def trial_duration(self) -> float:
        """Total duration of one trial in seconds."""
        return self.rest_duration + self.cue_duration + self.mi_duration 
    
    @property
    def mi_start_offset(self) -> float:
        """Start of MI period relative to trial start (seconds)."""
        return self.rest_duration + self.cue_duration
    
    @property
    def mi_end_offset(self) -> float:
        """End of MI period relative to trial start (seconds)."""
        return self.mi_start_offset + self.mi_duration
    
    @property
    def mi_samples_raw(self) -> int:
        """Number of raw samples in MI period."""
        return int(self.mi_duration * self.raw_sample_rate)
    
    @property
    def mi_samples_processed(self) -> int:
        """Number of processed samples in MI period."""
        return int(self.mi_duration * self.processed_sample_rate)


@dataclass 
class TrainingConfig:
    """Configuration for the training pipeline."""
    
    # Trial configuration
    trial_config: TrialConfig = None
    
    # CSP configuration
    n_csp_components: int = 4
    csp_regularization: str = 'ledoit_wolf'
    
    # Train/test split
    test_size: float = 0.2
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Verbose output
    verbose: bool = True
    
    def __post_init__(self):
        if self.trial_config is None:
            self.trial_config = TrialConfig()


# =============================================================================
# Data Loading
# =============================================================================

def parse_cue_file(cue_filepath: str) -> List[str]:
    """
    Parse the cue order file.
    
    Parameters
    ----------
    cue_filepath : str
        Path to the txt file with one cue per line.
        
    Returns
    -------
    cues : list of str
        List of cue labels ('Thumb', 'Index', 'Pinky', 'Rest')
        Note: 'Rest' cues are included in the list but will be filtered
              out during trial segmentation (not used for training).
    """
    cues = []
    with open(cue_filepath, 'r') as f:
        for line in f:
            cue = line.strip()
            if cue:
                if cue.lower() in ['thumb', 'index', 'pinky', 'rest']:
                    cues.append(cue.capitalize())
                
    return cues


def load_raw_eeg_data(csv_filepath: str) -> pd.DataFrame:
    """
    Load raw EEG data from CSV file.
    
    Parameters
    ----------
    csv_filepath : str
        Path to the CSV file from 'collect' mode.
        
    Returns
    -------
    df : DataFrame
        Raw EEG data with Time and channel columns.
    """
    df = pd.read_csv(csv_filepath)
    
    # Ensure Time column exists
    if 'Time' not in df.columns:
        raise ValueError("CSV must have 'Time' column")
    
    # Sort by time
    df = df.sort_values('Time').reset_index(drop=True)
    
    return df


def segment_trials(
    df: pd.DataFrame,
    cues: List[str],
    config: TrialConfig,
    start_time: Optional[float] = None
) -> List[Dict]:
    """
    Segment raw EEG data into trials based on trial structure.
    
    Trials with cue='Rest' are SKIPPED and not included in output.
    
    Parameters
    ----------
    df : DataFrame
        Raw EEG data.
    cues : list of str
        Ordered list of cue labels for each trial.
        Can include 'Rest' which will be skipped.
    config : TrialConfig
        Trial timing configuration.
    start_time : float, optional
        Start time of first trial. If None, uses first sample time.
        
    Returns
    -------
    trials : list of dict
        List of trial dictionaries with 'data' (MI period) and 'label'.
        'Rest' trials are excluded from this list.
    """
    if start_time is None: 
        start_time = df['Time'].iloc[0]
    
    eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    
    trials = []
    skipped_rest_count = 0
    
    for trial_idx, cue in enumerate(cues):
        # Skip Rest trials - they are recorded but not used for training
        if cue.lower() == 'rest':
            skipped_rest_count += 1
            continue
        
        # Calculate trial boundaries
        trial_start = start_time + trial_idx * config.trial_duration
        mi_start = trial_start + config.mi_start_offset
        mi_end = trial_start + config.mi_end_offset
        
        # Extract MI period data
        mask = (df['Time'] >= mi_start) & (df['Time'] < mi_end)
        trial_data = df.loc[mask, ['Time'] + eeg_channels].copy()
        
        if len(trial_data) == 0:
            print(f"Warning: Trial {trial_idx} ({cue}) has no data (time: {mi_start:.2f}-{mi_end:.2f})")
            continue
        
        # Convert label string to index (only for non-Rest classes)
        label_idx = config.class_labels.index(cue)
        
        trials.append({
            'trial_idx': trial_idx,
            'data': trial_data,
            'label': label_idx,
            'label_name': cue,
            'mi_start': mi_start,
            'mi_end': mi_end,
            'n_samples': len(trial_data)
        })
    
    if skipped_rest_count > 0:
        print(f"  Skipped {skipped_rest_count} 'Rest' trials (not used for training)")
    
    return trials


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_trials(
    trials: List[Dict],
    verbose: bool = True
) -> List[Dict]:
    """
    Apply preprocessing to each trial's MI data.
    
    Parameters
    ----------
    trials : list of dict
        Raw trial data from segment_trials().
    verbose : bool
        Print progress.
        
    Returns
    -------
    processed_trials : list of dict
        Trials with preprocessed multiband data.
    """
    preprocessor = EEGPreprocessor()
    
    processed_trials = []
    skipped_short_trials = 0
    
    for i, trial in enumerate(trials):
        if verbose and i % 50 == 0:
            print(f"  Preprocessing trial {i+1}/{len(trials)}...")
        
        # Check if trial has enough samples for filtering (need at least 28 samples)
        # The filter requires padlen=27, so we need more than that
        if len(trial['data']) < 30:
            if verbose:
                print(f"  Warning: Trial {trial['trial_idx']} ({trial['label_name']}) "
                      f"has only {len(trial['data'])} samples, skipping (need >= 30)")
            skipped_short_trials += 1
            continue
        
        try:
            # Process batch with multiband output
            processed = preprocessor.process_batch_multiband(
                trial['data'], 
                include_time=False
            )
            
            # processed shape: (n_samples_downsampled, 16) [8 alpha + 8 beta]
            processed_trials.append({
                'trial_idx': trial['trial_idx'],
                'data': processed,  # numpy array (n_samples, 16)
                'label': trial['label'],
                'label_name': trial['label_name']
            })
        except ValueError as e:
            if "padlen" in str(e):
                if verbose:
                    print(f"  Warning: Trial {trial['trial_idx']} ({trial['label_name']}) "
                          f"failed filtering (too short), skipping")
                skipped_short_trials += 1
            else:
                raise
    
    if skipped_short_trials > 0 and verbose:
        print(f"  Skipped {skipped_short_trials} trials due to insufficient data length")
    
    return processed_trials


# =============================================================================
# Windowing
# =============================================================================

def create_windows_from_trials(
    trials: List[Dict],
    window_config: Optional[WindowConfig] = None,
    verbose: bool = True
) -> Tuple[List[LabeledWindow], np.ndarray]:
    """
    Create sliding windows from preprocessed trials.
    
    Parameters
    ----------
    trials : list of dict
        Preprocessed trials with multiband data.
    window_config : WindowConfig, optional
        Window configuration. Uses defaults if not provided.
    verbose : bool
        Print progress.
        
    Returns
    -------
    windows : list of LabeledWindow
        All extracted windows.
    labels : ndarray
        Labels for each window.
    """
    if window_config is None:
        window_config = WindowConfig(use_multiband=True)
    
    generator = SlidingWindowGenerator(config=window_config)
    
    all_windows = []
    
    for trial in trials:
        # trial['data'] shape: (n_samples, 16) for multiband
        windows = generator.process_trial(
            trial_data=trial['data'],
            label=trial['label'],
            trial_id=trial['trial_idx']
        )
        all_windows.extend(windows)
    
    if verbose:
        print(f"  Created {len(all_windows)} windows from {len(trials)} trials")
    
    # Extract labels
    labels = np.array([w.label for w in all_windows])
    
    return all_windows, labels


# =============================================================================
# Training Pipeline
# =============================================================================

class TrainingPipeline:
    """
    End-to-end training pipeline for CSP + LDA classifier.
    
    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.csp_model: Optional[DualBandPairwiseCSP] = None
        self.lda_classifier: Optional[CSPLDAClassifier] = None
        self.normalizer: Optional[SessionNormalizer] = None
        
        # Training data statistics
        self.n_trials = 0
        self.n_windows = 0
        self.class_distribution = {}
    
    def load_data(
        self,
        csv_filepath: str,
        cue_filepath: str,
        start_time: Optional[float] = None
    ) -> Tuple[List[Dict], List[str]]:
        """
        Load and segment raw EEG data.
        
        Parameters
        ----------
        csv_filepath : str
            Path to raw EEG CSV file.
        cue_filepath : str
            Path to cue order txt file.
        start_time : float, optional
            Override start time for trial segmentation.
            
        Returns
        -------
        trials : list of dict
            Segmented trial data.
        cues : list of str
            Cue labels.
        """
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Loading Data")
            print("=" * 60)
        
        # Load cue order
        cues = parse_cue_file(cue_filepath)
        if self.config.verbose:
            print(f"  Loaded {len(cues)} cues from {cue_filepath}")
            print(f"  Class distribution: {dict(pd.Series(cues).value_counts())}")
        
        # Load raw EEG
        df = load_raw_eeg_data(csv_filepath)
        if self.config.verbose:
            duration = df['Time'].max() - df['Time'].min()
            print(f"  Loaded {len(df)} samples ({duration:.1f}s) from {csv_filepath}")
        
        # Segment into trials
        trials = segment_trials(df, cues, self.config.trial_config, start_time)
        if self.config.verbose:
            print(f"  Segmented {len(trials)} trials")
        
        self.n_trials = len(trials)
        
        return trials, cues
    
    def preprocess(self, trials: List[Dict]) -> List[Dict]:
        """Preprocess trials (filtering, downsampling, multiband)."""
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Preprocessing")
            print("=" * 60)
        
        processed = preprocess_trials(trials, verbose=self.config.verbose)
        
        if self.config.verbose:
            sample_shape = processed[0]['data'].shape if processed else None
            print(f"  Preprocessed {len(processed)} trials")
            print(f"  Sample shape: {sample_shape} (samples, channels)")
        
        return processed
    
    def create_windows(
        self, 
        trials: List[Dict]
    ) -> Tuple[List[LabeledWindow], np.ndarray]:
        """Create sliding windows from preprocessed trials."""
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Windowing")
            print("=" * 60)
        
        window_config = WindowConfig(use_multiband=True)
        windows, labels = create_windows_from_trials(
            trials, 
            window_config, 
            verbose=self.config.verbose
        )
        
        self.n_windows = len(windows)
        self.class_distribution = dict(pd.Series(labels).value_counts().sort_index())
        
        if self.config.verbose:
            print(f"  Window config: {window_config.window_samples} samples, "
                  f"{window_config.step_samples} step")
            print(f"  Class distribution: {self.class_distribution}")
        
        return windows, labels
    
    def train_csp(
        self,
        windows: List[LabeledWindow],
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train CSP model and extract features.
        
        Parameters
        ----------
        windows : list of LabeledWindow
            Training windows.
        labels : ndarray
            Window labels.
            
        Returns
        -------
        X_features : ndarray
            CSP features (n_windows, 24).
        y : ndarray
            Labels.
        """
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Training CSP")
            print("=" * 60)
        
        # Prepare dual-band data
        X_alpha, X_beta, y = prepare_dual_band_data(windows, n_channels_per_band=8)
        
        if self.config.verbose:
            print(f"  Alpha data shape: {X_alpha.shape}")
            print(f"  Beta data shape: {X_beta.shape}")
        
        # Create and fit CSP model
        csp_config = DualBandCSPConfig(
            n_components=self.config.n_csp_components,
            reg=self.config.csp_regularization
        )
        
        self.csp_model = DualBandPairwiseCSP(config=csp_config)
        self.csp_model.fit(X_alpha, X_beta, y)
        
        if self.config.verbose:
            print(f"  CSP fitted with {csp_config.n_components} components per pair")
            print(f"  Total features: {self.csp_model.config.total_features}")
        
        # Extract features
        X_features = self.csp_model.transform(X_alpha, X_beta)
        
        if self.config.verbose:
            print(f"  Feature matrix shape: {X_features.shape}")
        
        return X_features, y
    
    def train_normalizer(self, X_features: np.ndarray) -> np.ndarray:
        """
        Fit session normalizer and transform features.
        
        Parameters
        ----------
        X_features : ndarray
            Raw CSP features.
            
        Returns
        -------
        X_normalized : ndarray
            Z-scored features.
        """
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Session Normalization")
            print("=" * 60)
        
        self.normalizer = SessionNormalizer()
        X_normalized = self.normalizer.fit_transform(X_features)
        
        if self.config.verbose:
            print(f"  Feature mean (before): {X_features.mean(axis=0)[:4]}...")
            print(f"  Feature std (before): {X_features.std(axis=0)[:4]}...")
            print(f"  Feature mean (after): {X_normalized.mean(axis=0)[:4]}...")
            print(f"  Feature std (after): {X_normalized.std(axis=0)[:4]}...")
        
        return X_normalized
    
    def train_lda(
        self,
        X_features: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Train LDA classifier and return training accuracy.
        
        Parameters
        ----------
        X_features : ndarray
            Normalized CSP features.
        y : ndarray
            Labels.
            
        Returns
        -------
        accuracy : float
            Training accuracy.
        """
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Training LDA Classifier")
            print("=" * 60)
        
        # Convert numeric labels to string labels
        label_names = [self.config.trial_config.class_labels[i] for i in y]
        
        self.lda_classifier = CSPLDAClassifier()
        self.lda_classifier.fit(X_features, np.array(label_names))
        
        accuracy = self.lda_classifier.score(X_features, np.array(label_names))
        
        if self.config.verbose:
            print(f"  Training accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def cross_validate(
        self,
        X_features: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5
    ) -> Dict:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X_features : ndarray
            Features.
        y : ndarray
            Labels.
        n_folds : int
            Number of CV folds.
            
        Returns
        -------
        results : dict
            Cross-validation results.
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print(f"{n_folds}-Fold Cross-Validation")
            print("=" * 60)
        
        lda = LinearDiscriminantAnalysis(solver='svd')
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_seed)
        
        scores = cross_val_score(lda, X_features, y, cv=cv)
        
        results = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'fold_scores': scores.tolist()
        }
        
        if self.config.verbose:
            print(f"  Fold accuracies: {[f'{s:.2%}' for s in scores]}")
            print(f"  Mean accuracy: {results['mean_accuracy']:.2%} ± {results['std_accuracy']:.2%}")
        
        return results
    
    def save_models(self, output_dir: str) -> None:
        """
        Save trained models to disk.
        
        Parameters
        ----------
        output_dir : str
            Directory to save models.
        """
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Saving Models")
            print("=" * 60)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSP model
        csp_path = output_path / "csp_model.pkl"
        self.csp_model.save(csp_path)
        if self.config.verbose:
            print(f"  Saved CSP model to {csp_path}")
        
        # Save LDA classifier
        lda_path = output_path / "lda_classifier.pkl"
        self.lda_classifier.save(lda_path)
        if self.config.verbose:
            print(f"  Saved LDA classifier to {lda_path}")
        
        # Save normalizer
        normalizer_path = output_path / "normalizer.pkl"
        self.normalizer.save(normalizer_path)
        if self.config.verbose:
            print(f"  Saved normalizer to {normalizer_path}")
        
        # Save training metadata
        import json
        metadata = {
            'n_trials': int(self.n_trials),
            'n_windows': int(self.n_windows),
            'class_distribution': {str(k): int(v) for k, v in self.class_distribution.items()},
            'n_csp_components': int(self.config.n_csp_components),
            'csp_regularization': self.config.csp_regularization,
            'class_labels': self.config.trial_config.class_labels,
            'n_features': int(self.csp_model.config.total_features)
        }
        
        metadata_path = output_path / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        if self.config.verbose:
            print(f"  Saved metadata to {metadata_path}")
    
    def run(
        self,
        csv_filepath: str,
        cue_filepath: str,
        output_dir: str,
        start_time: Optional[float] = None,
        run_cv: bool = True
    ) -> Dict:
        """
        Run the full training pipeline.
        
        Parameters
        ----------
        csv_filepath : str
            Path to raw EEG CSV file.
        cue_filepath : str
            Path to cue order txt file.
        output_dir : str
            Directory to save trained models.
        start_time : float, optional
            Override start time for trial segmentation.
        run_cv : bool
            Whether to run cross-validation.
            
        Returns
        -------
        results : dict
            Training results and statistics.
        """
        # 1. Load data
        trials, cues = self.load_data(csv_filepath, cue_filepath, start_time)
        
        # 2. Preprocess
        processed_trials = self.preprocess(trials)
        
        # 3. Create windows
        windows, labels = self.create_windows(processed_trials)
        
        # 4. Train CSP
        X_features, y = self.train_csp(windows, labels)
        
        # 5. Normalize features
        X_normalized = self.train_normalizer(X_features)
        
        # 6. Cross-validation (optional)
        cv_results = None
        if run_cv:
            cv_results = self.cross_validate(X_normalized, y)
        
        # 7. Train final LDA on all data
        train_accuracy = self.train_lda(X_normalized, y)
        
        # 8. Save models
        self.save_models(output_dir)
        
        # Compile results
        results = {
            'n_trials': self.n_trials,
            'n_windows': self.n_windows,
            'class_distribution': self.class_distribution,
            'n_features': self.csp_model.config.total_features,
            'train_accuracy': train_accuracy,
            'cv_results': cv_results
        }
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Training Complete!")
            print("=" * 60)
            print(f"  Trials: {self.n_trials}")
            print(f"  Windows: {self.n_windows}")
            print(f"  Features: {self.csp_model.config.total_features}")
            print(f"  Training accuracy: {train_accuracy:.2%}")
            if cv_results:
                print(f"  CV accuracy: {cv_results['mean_accuracy']:.2%} ± {cv_results['std_accuracy']:.2%}")
            print(f"  Models saved to: {output_dir}")
        
        return results


# =============================================================================
# CLI
# =============================================================================

def train_model(
    csv_path: str,
    cue_path: str,
    output_dir: str,
    n_components: int = 4,
    start_time: Optional[float] = None,
    run_cv: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Train CSP + LDA model from collected data.
    
    Parameters
    ----------
    csv_path : str
        Path to raw EEG CSV file.
    cue_path : str
        Path to cue order txt file.
    output_dir : str
        Directory to save trained models.
    n_components : int
        Number of CSP components per pair.
    start_time : float, optional
        Override trial start time.
    run_cv : bool
        Run cross-validation.
    verbose : bool
        Print progress.
        
    Returns
    -------
    results : dict
        Training results.
    """
    config = TrainingConfig(
        n_csp_components=n_components,
        verbose=verbose
    )
    
    pipeline = TrainingPipeline(config)
    results = pipeline.run(
        csv_filepath=csv_path,
        cue_filepath=cue_path,
        output_dir=output_dir,
        start_time=start_time,
        run_cv=run_cv
    )
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CSP + LDA classifier from collected EEG data"
    )
    
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to raw EEG CSV file from collect mode"
    )
    parser.add_argument(
        "--cues", type=str, required=True,
        help="Path to cue order txt file"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--n-components", type=int, default=4,
        help="Number of CSP components per pair (default: 4)"
    )
    parser.add_argument(
        "--start-time", type=float, default=None,
        help="Override trial start time (default: first sample time)"
    )
    parser.add_argument(
        "--no-cv", action="store_true",
        help="Skip cross-validation"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    results = train_model(
        csv_path=args.csv,
        cue_path=args.cues,
        output_dir=args.output,
        n_components=args.n_components,
        start_time=args.start_time,
        run_cv=not args.no_cv,
        verbose=not args.quiet
    )
    
    print("\nDone!")
