import os
from random import sample
import subprocess
import argparse
import string
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Callable

import unicorneeg.stream as ulsl
from unicorneeg.clean import (
    EEGPreprocessor, 
    create_preprocessing_callback,
    create_multiband_preprocessing_callback
)
from unicorneeg.pipe import (
    RealTimeWindowBuffer, 
    WindowConfig, 
    create_windowing_callback,
    LabeledWindow
)
from control.csp_features import (
    DualBandPairwiseCSP, 
    DualBandRealTimeCSPExtractor,
    SessionNormalizer
)
from control.lda_classifier import CSPLDAClassifier, OnlineClassifier, ClassificationResult
from control.lda_train import train_model, TrainingPipeline, TrainingConfig


# =============================================================================
# Online Classification Pipeline
# =============================================================================

class OnlineClassificationPipeline:
    """
    End-to-end online classification pipeline for EEG motor imagery.
    
    Combines: preprocessing → multiband windowing → CSP features → LDA classification
    
    Parameters
    ----------
    model_path : str or Path
        Path to the model directory containing:
        - csp_model.pkl: Fitted DualBandPairwiseCSP model
        - lda_classifier.pkl: Fitted CSPLDAClassifier
        - normalizer.pkl: Session normalizer (optional)
    smoothing_method : str
        'ema' for exponential moving average, 'none' for raw predictions
    smoothing_alpha : float
        EMA smoothing factor (0-1). Higher = more weight on recent.
    """
    
    def __init__(
        self,
        model_path: str,
        smoothing_method: str = 'ema',
        smoothing_alpha: float = 0.3
    ):
        self.model_path = Path(model_path)
        self.smoothing_method = smoothing_method
        self.smoothing_alpha = smoothing_alpha
        
        # Load models
        self._load_models()
        
        # Initialize components
        self.csp_extractor = DualBandRealTimeCSPExtractor(self.csp_model)
        
        if smoothing_method != 'none':
            self.online_classifier = OnlineClassifier(
                self.lda_classifier,
                smoothing_method=smoothing_method,
                smoothing_alpha=smoothing_alpha
            )
        else:
            self.online_classifier = None
        
        # Callback for classification results
        self._on_classification_callback: Optional[Callable[[ClassificationResult], None]] = None
        
        # Statistics
        self.n_windows_processed = 0
        self.last_result: Optional[ClassificationResult] = None
        
    def _load_models(self):
        """Load pre-trained models from disk."""
        csp_path = self.model_path / "csp_model.pkl"
        lda_path = self.model_path / "lda_classifier.pkl"
        normalizer_path = self.model_path / "normalizer.pkl"
        
        if not csp_path.exists():
            raise FileNotFoundError(f"CSP model not found at {csp_path}")
        if not lda_path.exists():
            raise FileNotFoundError(f"LDA classifier not found at {lda_path}")
        
        print(f"[Pipeline] Loading CSP model from {csp_path}")
        self.csp_model = DualBandPairwiseCSP.load(csp_path)
        
        print(f"[Pipeline] Loading LDA classifier from {lda_path}")
        self.lda_classifier = CSPLDAClassifier.load(lda_path)
        
        if normalizer_path.exists():
            print(f"[Pipeline] Loading session normalizer from {normalizer_path}")
            self.normalizer = SessionNormalizer.load(normalizer_path)
        else:
            print("[Pipeline] No normalizer found, using raw features")
            self.normalizer = None
    
    def on_classification(self, callback: Callable[[ClassificationResult], None]) -> 'OnlineClassificationPipeline':
        """
        Register callback for classification results.
        
        Parameters
        ----------
        callback : callable
            Function to call with ClassificationResult for each window.
        """
        self._on_classification_callback = callback
        return self
    
    def process_window(self, window: LabeledWindow) -> ClassificationResult:
        """
        Process a single multiband window through the full pipeline.
        
        Parameters
        ----------
        window : LabeledWindow
            Window from RealTimeWindowBuffer (with use_multiband=True).
            
        Returns
        -------
        result : ClassificationResult
            Classification result with probabilities.
        """
        # 1. Extract CSP features (24-dimensional)
        features = self.csp_extractor.extract_from_multiband_window(window.data)
        
        # 2. Apply session normalization (z-score)
        if self.normalizer is not None:
            features = self.normalizer.transform(features.reshape(1, -1)).squeeze()
        
        # 3. Classify with optional smoothing
        if self.online_classifier is not None:
            result = self.online_classifier.update(features)
        else:
            result = self.lda_classifier.classify(features)
        
        # Update statistics
        self.n_windows_processed += 1
        self.last_result = result
        
        # Fire callback if registered
        if self._on_classification_callback is not None:
            self._on_classification_callback(result)
        
        return result
    
    def reset(self):
        """Reset smoothing state and counters."""
        if self.online_classifier is not None:
            self.online_classifier.reset()
        self.n_windows_processed = 0
        self.last_result = None
    
    def create_window_handler(self) -> Callable[[LabeledWindow], None]:
        """
        Create a window handler callback for use with RealTimeWindowBuffer.
        
        Returns
        -------
        callable
            Handler function that processes windows and prints results.
        """
        def handle_window(window: LabeledWindow):
            result = self.process_window(window)
            probs_str = ", ".join(
                f"{k}: {v:.2f}" for k, v in result.class_probabilities.items()
            )
            print(f"[Window {self.n_windows_processed}] {result.predicted_class} | {probs_str}")
        
        return handle_window


def run_online_classification(
    model_path: str,
    graphing: bool = True,
    duration: int = 0,
    run_name: str = "",
    burn_in_seconds: float = 15.0,
    smoothing_method: str = 'ema',
    smoothing_alpha: float = 0.3
):
    """
    Run online EEG classification with pre-trained models.
    
    Pipeline: Raw EEG → Preprocessing (multiband) → Windowing → CSP → LDA → Probabilities
    
    Parameters
    ----------
    model_path : str
        Path to model directory containing csp_model.pkl and lda_classifier.pkl
    graphing : bool
        Whether to show real-time EEG graph
    duration : int
        Recording duration in seconds (0 = indefinite)
    run_name : str
        Name for logging
    burn_in_seconds : float
        Burn-in period where data is processed but not classified
    smoothing_method : str
        'ema' for exponential moving average, 'none' for raw
    smoothing_alpha : float
        EMA smoothing factor
    """
    
    # Create classification pipeline
    print(f"\n[Online Mode] Loading models from: {model_path}")
    pipeline = OnlineClassificationPipeline(
        model_path=model_path,
        smoothing_method=smoothing_method,
        smoothing_alpha=smoothing_alpha
    )
    
    # Setup preprocessing (multiband: alpha + beta)
    preprocessor = EEGPreprocessor()
    
    # Setup windowing (multiband: 16 channels = 8 alpha + 8 beta)
    window_config = WindowConfig(use_multiband=True)
    window_buffer = RealTimeWindowBuffer(config=window_config, current_label=0)
    
    # Register classification handler
    window_buffer.on_window(pipeline.create_window_handler())
    
    # Create callback chain: raw → preprocess (multiband) → window → classify
    windowing_callback = create_windowing_callback(window_buffer)
    preprocessing_callback = create_multiband_preprocessing_callback(
        preprocessor=preprocessor,
        output_callback=windowing_callback
    )
    
    # Setup EEG stream with visualization console
    config = ulsl.EEGStreamConfig(
        use_background_thread=True,
        enable_visualization=graphing,
        enable_graphing=graphing,  # Legacy compat
        save_duration_seconds=duration,
        csv_path=os.path.join("data/streamlogs", run_name + "_eeg_data.csv") if run_name else None,
        burn_in_seconds=burn_in_seconds
    )
    
    stream = ulsl.EEGStream(config)
    stream.on_sample(preprocessing_callback)
    
    # Register burn-in complete callback
    def on_burn_in_done():
        print("[Burn-in] Complete. Starting classification...")
        window_buffer.reset()
        pipeline.reset()
    
    stream.on_burn_in_complete(on_burn_in_done)
    
    # Start
    input("Press Enter to start online classification...")
    
    print("\nConnecting to EEG stream...")
    try:
        stream.connect()
        print("Connected! Starting online classification...")
        print(f"Window: {window_config.window_samples} samples ({window_config.window_length_ms}ms)")
        print(f"Step: {window_config.step_samples} samples ({window_config.step_size_ms}ms)")
        print(f"Channels: {window_config.n_channels} (multiband: 8 alpha + 8 beta)")
        print(f"Features: 24 (CSP dual-band)")
        print("-" * 60)
        stream.start()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Make sure the Unicorn EEG device is streaming via LSL.")
        sys.exit(1)
    
    return pipeline

def collect_data(
    background_prcs: bool = False, 
    visualization: bool = True, 
    duration: int = 0, 
    run_name: str = "", 
    current_label: int = 0, 
    burn_in_seconds: float = 15.0,
    use_multiband: bool = False
):
    """
    Collect EEG data with preprocessing, windowing, and visualization console.
    
    Parameters
    ----------
    background_prcs : bool
        Whether to run stream in background thread
    visualization : bool
        Whether to show real-time visualization console
    duration : int
        Recording duration in seconds (0 = indefinite)
    run_name : str
        Name for saving CSV log
    current_label : int
        Label for windows (e.g., 0=Thumb, 1=Index, 2=Pinky)
    burn_in_seconds : float
        Burn-in period where data is discarded
    use_multiband : bool
        If True, output alpha and beta bands separately (16 channels).
        If False, output combined band (8 channels).
    """
    
    def handle_window(window):
        """Called when a new window is ready (94 samples, 750ms)."""
        print(f"Window {window.window_idx}: label={window.label}, shape={window.data.shape}")
        pass
    
    # Setup preprocessing pipeline
    preprocessor = EEGPreprocessor()
    
    # Setup windowing pipeline (750ms window, 125ms step)
    window_config = WindowConfig(use_multiband=use_multiband)
    window_buffer = RealTimeWindowBuffer(config=window_config, current_label=current_label)
    window_buffer.on_window(handle_window)
    
    # Create chained callbacks: raw -> preprocess -> window
    windowing_callback = create_windowing_callback(window_buffer)
    
    if use_multiband:
        preprocessing_callback = create_multiband_preprocessing_callback(
            preprocessor=preprocessor,
            output_callback=windowing_callback
        )
    else:
        preprocessing_callback = create_preprocessing_callback(
            preprocessor=preprocessor,
            output_callback=windowing_callback,
            output_band='combined'
        )

    config = ulsl.EEGStreamConfig(
        use_background_thread=background_prcs,
        enable_visualization=visualization,
        enable_graphing=visualization,  # Legacy compat
        save_duration_seconds=duration,
        csv_path=os.path.join("data/streamlogs", run_name + "_eeg_data.csv"),
        burn_in_seconds=burn_in_seconds
    )
    
    stream = ulsl.EEGStream(config)
    stream.on_sample(preprocessing_callback)
    
    # Register burn-in complete callback to reset window buffer
    def on_burn_in_done():
        print("[Burn-in] Resetting window buffer...")
        window_buffer.reset()
    
    stream.on_burn_in_complete(on_burn_in_done)
    
    usrconfirm = input("Press Enter to start data collection...")
    
    print("Connecting to EEG stream...")
    try:
        stream.connect()
        print("Connected! Starting data collection...")
        print(f"Windowing: {window_config.window_samples} samples ({window_config.window_length_ms}ms), "
              f"step: {window_config.step_samples} samples ({window_config.step_size_ms}ms)")
        print(f"Multiband: {use_multiband} ({window_config.n_channels} channels)")
        stream.start()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Make sure the Unicorn EEG device is streaming via LSL.")
        sys.exit(1)
    
    return window_buffer  # Return buffer for access to collected windows


if __name__ == "__main__":
    header = '''
================================================================================================================
 ________ _____   __   _         __    ___               __  __       __  _       _____          __           __
/_  __/ // / _ | / /  (_)__     / /   / _ \_______  ___ / /_/ /  ___ / /_(_)___  / ___/__  ___  / /________  / /
 / / / _  / __ |/ /__/ (_-<    / /   / ___/ __/ _ \(_-</ __/ _ \/ -_) __/ / __/ / /__/ _ \/ _ \/ __/ __/ _ \/ / 
/_/ /_//_/_/ |_/____/_/___/   / /   /_/  /_/  \___/___/\__/_//_/\__/\__/_/\__/  \___/\___/_//_/\__/_/  \___/_/  
                             /_/                                                                                
EEG-Based Prosthetic Control System v.0.1.0                            
Written by Krishna Bhatt (krishbhatt2019@gmail.com)
Latest version: https://github.com/blizzard-labs/thalis-eeg-control                         
================================================================================================================
    '''
    print(header)
    
    parser = argparse.ArgumentParser(description="EEG-Based Prosthetic Control")
    characters = string.ascii_letters + string.digits
    
    help_msg = '''
        Thalamic Integration System: EEG-Based Prosthetic Control
        Usage: python main.py --mode <mode> --model <model_path> --name <run_name> [OPTIONS]
        
        Modes:
          default  - Online classification using pre-trained model
          collect  - Collect labeled EEG data for training
          train    - Train CSP + LDA model from collected data
          simulate - Run hand simulation (requires model)
          fine-tune - Fine-tune existing model with new data
    '''
    
    parser.add_argument("--mode", type=str, 
                        choices=['default', 'collect', 'simulate', 'fine-tune', 'train'],
                        required=True, 
                        help="Operation mode: 'default' (online), 'collect', 'train', 'simulate', 'fine-tune'")
    parser.add_argument("--model", type=str, required=False, default="models/default",
                        help="Path to model directory (default: models/default)")
    parser.add_argument("--name", type=str, required=False, default="run",
                        help="Name for this run (used for logging)")
    
    parser.add_argument("--duration", type=int, default=100,
                        required=False, 
                        help="Duration for data collection in seconds (default: 100s)")
    
    parser.add_argument("--burn-in", type=float, default=15.0,
                        required=False, 
                        help="Burn-in period in seconds (default: 15s)")
    
    parser.add_argument("--label", type=int, default=0,
                        required=False,
                        help="Label for data collection: 0=Thumb, 1=Index, 2=Pinky")
    
    parser.add_argument("--smoothing", type=str, default='ema',
                        choices=['ema', 'none'],
                        help="Prediction smoothing method (default: ema)")
    
    parser.add_argument("--smoothing-alpha", type=float, default=0.3,
                        help="EMA smoothing factor 0-1 (default: 0.3)")
    
    parser.add_argument("--multiband", action='store_true',
                        help="Use multiband (alpha+beta) preprocessing for collect mode")
    
    parser.add_argument("--csv", type=str, required=False,
                        help="Path to raw EEG CSV file (for train mode)")
    parser.add_argument("--cues", type=str, required=False,
                        help="Path to cue order txt file (for train mode)")
    parser.add_argument("--n-components", type=int, default=4,
                        help="Number of CSP components per pair (for train mode)")
    parser.add_argument("--start-time", type=float, default=None,
                        help="Override trial start time in seconds (for train mode)")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip cross-validation (for train mode)")
    
    args = parser.parse_args()
    print(f"Selected Mode: {args.mode}")
    
    
    if args.mode == 'default':
        '''
        DEFAULT MODE: Online classification with pre-trained CSP + LDA model.
        Pipeline: EEG → Preprocessing → Windowing → CSP features → LDA → Probabilities
        '''
        run_online_classification(
            model_path=args.model,
            graphing=True,
            duration=args.duration,
            run_name=args.name,
            burn_in_seconds=args.burn_in,
            smoothing_method=args.smoothing,
            smoothing_alpha=args.smoothing_alpha
        )
        
    elif args.mode == 'collect':
        '''
        COLLECT MODE: Collect labeled EEG data for training.
        Use --label to set the class (0=Thumb, 1=Index, 2=Pinky).
        Use --multiband for dual-band features.
        
        Launches the EEG Visualization Console with:
        - Multi-channel time series display (raw/filtered toggle)
        - RMS amplitude heatmap
        - Signal quality indicators
        - Topographic band power map
        '''
        print(f"\n[Collect Mode] Label: {args.label} ({['Thumb', 'Index', 'Pinky'][args.label]})")
        print(f"[Collect Mode] Multiband: {args.multiband}")
        print("[Collect Mode] Launching visualization console...")
        collect_data(
            background_prcs=False, 
            visualization=True, 
            duration=args.duration, 
            run_name=args.name,
            current_label=args.label,
            burn_in_seconds=args.burn_in,
            use_multiband=args.multiband
        )
    
    elif args.mode == 'train':
        '''
        TRAIN MODE: Train CSP + LDA model from collected data.
        
        Required arguments:
          --csv   Path to raw EEG CSV file from collect mode
          --cues  Path to cue order txt file
          --model Output directory for trained models
          
        Optional arguments:
          --n-components   CSP components per pair (default: 4)
          --start-time     Override trial start time
          --no-cv          Skip cross-validation
        '''
        if not args.csv or not args.cues:
            print("\n[Train Mode] ERROR: --csv and --cues arguments are required for train mode")
            print("Usage: python main.py --mode train --csv <eeg_data.csv> --cues <cues.txt> --model <output_dir>")
            sys.exit(1)
        
        print(f"\n[Train Mode] Training CSP + LDA classifier")
        print(f"  CSV file: {args.csv}")
        print(f"  Cue file: {args.cues}")
        print(f"  Output: {args.model}")
        print(f"  CSP components: {args.n_components}")
        
        results = train_model(
            csv_path=args.csv,
            cue_path=args.cues,
            output_dir=args.model,
            n_components=args.n_components,
            start_time=args.start_time,
            run_cv=not args.no_cv,
            verbose=True
        )
    
    elif args.mode == 'simulate':
        '''
        SIMULATE MODE: Run prosthetic hand simulation.
        '''
        print("\n[Simulate Mode] Hand simulation not yet implemented.")
        # TODO: Integrate with handsim module
    
    elif args.mode == 'fine-tune':
        '''
        FINE-TUNE MODE: Fine-tune existing model with new data.
        '''
        print("\n[Fine-tune Mode] Fine-tuning not yet implemented.")
        # TODO: Implement fine-tuning