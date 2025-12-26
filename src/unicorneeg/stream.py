"""
EEG Stream Module - Object-oriented LSL stream handler for Unicorn EEG devices.

Provides configurable options for:
- Real-time graphing
- CSV data saving
- Background thread data collection for concurrent processing
"""

from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any
from pathlib import Path
import threading
import time


@dataclass
class EEGStreamConfig:
    """Configuration for EEGStream."""
    stream_name: str = 'UN-2024.06.42'
    sample_rate: int = 250  # Hz
    buffer_seconds: int = 10  # seconds of data to keep in rolling buffer
    
    # Column configuration
    columns: List[str] = field(default_factory=lambda: [
        'Time', 'FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8',
        'AccX', 'AccY', 'AccZ', 'Gyro1', 'Gyro2', 'Gyro3', 'Battery', 'Counter', 'Validation'
    ])
    
    # Graphing options
    enable_graphing: bool = True
    plot_rows: int = 5
    plot_cols: int = 4
    figure_size: tuple = (16, 12)
    
    # CSV saving options
    enable_csv_save: bool = True
    csv_path: str = 'EEGdata.csv'
    save_duration_seconds: float = 60.0
    
    # Threading options
    use_background_thread: bool = False
    
    # Burn-in period: data is processed but discarded during this time
    # Plots and buffers are reset after burn-in completes
    burn_in_seconds: float = 0.0  # 0 means no burn-in
    
    @property
    def buffer_maxlen(self) -> int:
        return self.sample_rate * self.buffer_seconds
    
    @property
    def plot_columns(self) -> List[str]:
        return [c for c in self.columns if c != 'Time']


class EEGStream:
    """
    Object-oriented EEG data stream handler with support for real-time
    graphing, CSV saving, and background thread data collection.
    
    Examples
    --------
    Basic usage with graphing and CSV saving:
    
        >>> stream = EEGStream()
        >>> stream.start()  # Blocks until duration complete, shows live plot
    
    Background collection with callback for concurrent processing:
    
        >>> def process_sample(sample: dict):
        ...     # Process each sample in real-time
        ...     print(f"Got sample at {sample['Time']}")
        ...
        >>> config = EEGStreamConfig(use_background_thread=True, enable_graphing=False)
        >>> stream = EEGStream(config)
        >>> stream.on_sample(process_sample)
        >>> stream.start()  # Returns immediately
        >>> # ... do other work while collecting ...
        >>> stream.stop()
        >>> data = stream.get_dataframe()
    
    Just collect data without graphing:
    
        >>> config = EEGStreamConfig(enable_graphing=False, save_duration_seconds=30)
        >>> stream = EEGStream(config)
        >>> stream.start()
        >>> df = stream.get_dataframe()
    """
    
    def __init__(self, config: Optional[EEGStreamConfig] = None):
        """
        Initialize EEGStream with the given configuration.
        
        Parameters
        ----------
        config : EEGStreamConfig, optional
            Configuration object. Uses defaults if not provided.
        """
        self.config = config or EEGStreamConfig()
        self._inlet: Optional[StreamInlet] = None
        self._data_buffers: Dict[str, deque] = {}
        self._start_time: Optional[float] = None
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callbacks for sample processing
        self._sample_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Burn-in state tracking
        self._burn_in_complete: bool = False
        self._burn_in_callbacks: List[Callable[[], None]] = []
        
        # Plotting state
        self._fig = None
        self._axes = None
        self._lines: Dict[str, Any] = {}
        
    def connect(self) -> 'EEGStream':
        """
        Connect to the LSL stream.
        
        Returns
        -------
        EEGStream
            Self for method chaining.
            
        Raises
        ------
        RuntimeError
            If no LSL stream is found with the configured name.
        """
        streams = resolve_byprop('name', self.config.stream_name)
        if not streams:
            raise RuntimeError(f"No LSL stream found with name '{self.config.stream_name}'")
        self._inlet = StreamInlet(streams[0])
        self._init_buffers()
        return self
    
    def _init_buffers(self) -> None:
        """Initialize data buffers for all columns."""
        maxlen = self.config.buffer_maxlen
        self._data_buffers = {col: deque(maxlen=maxlen) for col in self.config.columns}
    
    def on_sample(self, callback: Callable[[Dict[str, Any]], None]) -> 'EEGStream':
        """
        Register a callback to be called for each new sample.
        
        Useful for concurrent processing when running in background thread.
        
        Parameters
        ----------
        callback : callable
            Function that receives a dict with column names as keys and sample values.
            
        Returns
        -------
        EEGStream
            Self for method chaining.
        """
        self._sample_callbacks.append(callback)
        return self
    
    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a previously registered callback."""
        if callback in self._sample_callbacks:
            self._sample_callbacks.remove(callback)
    
    def on_burn_in_complete(self, callback: Callable[[], None]) -> 'EEGStream':
        """
        Register a callback to be called when burn-in period completes.
        
        Useful for resetting external buffers/state after burn-in.
        
        Parameters
        ----------
        callback : callable
            Function with no arguments, called once when burn-in completes.
            
        Returns
        -------
        EEGStream
            Self for method chaining.
        """
        self._burn_in_callbacks.append(callback)
        return self
    
    def is_burn_in_complete(self) -> bool:
        """Check if burn-in period has completed."""
        return self._burn_in_complete
    
    def _setup_plot(self) -> None:
        """Initialize matplotlib figure and axes for real-time plotting."""
        if not self.config.enable_graphing:
            return
            
        plot_columns = self.config.plot_columns
        n_plots = len(plot_columns)
        rows, cols = self.config.plot_rows, self.config.plot_cols
        
        self._fig, self._axes = plt.subplots(
            rows, cols, 
            figsize=self.config.figure_size, 
            sharex=True
        )
        axes_flat = self._axes.flatten()
        
        self._lines = {}
        for i, col in enumerate(plot_columns):
            ax = axes_flat[i]
            self._lines[col], = ax.plot([], [], lw=1)
            ax.set_title(col)
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for j in range(n_plots, rows * cols):
            self._fig.delaxes(axes_flat[j])
        
        plt.tight_layout()
    
    def reset_plots(self) -> None:
        """
        Reset all plot lines to empty, clearing displayed data.
        
        Called automatically after burn-in period completes to give a fresh start.
        """
        if not self.config.enable_graphing or self._fig is None:
            return
            
        for col in self.config.plot_columns:
            if col in self._lines:
                self._lines[col].set_data([], [])
                ax = self._lines[col].axes
                ax.relim()
                ax.autoscale_view()
        
        plt.draw()
    
    def _update_plot(self) -> None:
        """Update plot with current buffer contents."""
        if not self.config.enable_graphing or self._fig is None:
            return
            
        with self._lock:
            t = list(self._data_buffers['Time'])
            if not t:
                return
            t0 = t[0]
            t_rel = [ti - t0 for ti in t]
            
            for col in self.config.plot_columns:
                y = list(self._data_buffers[col])
                self._lines[col].set_data(t_rel, y)
                ax = self._lines[col].axes
                ax.relim()
                ax.autoscale_view()
                
                #ax.set_xlim(left=0, right=max(t_rel) if t_rel else self.config.buffer_seconds)
                
                # Avoid singular xlims (left == right)
                max_t = max(t_rel) if t_rel else 0
                right_lim = max_t if max_t > 0 else self.config.buffer_seconds
                ax.set_xlim(left=0, right=right_lim)
                
        
        plt.pause(0.001)
    
    def _collect_sample(self) -> Optional[Dict[str, Any]]:
        """
        Pull and store a single sample from the stream.
        
        Returns
        -------
        dict or None
            The sample as a dict, or None if no sample available.
        """
        if self._inlet is None:
            return None
            
        data, timestamp = self._inlet.pull_sample(timeout=0.1)
        if data is None:
            return None
            
        if self._start_time is None:
            self._start_time = time.time()
        
        all_data = [timestamp] + data
        sample = {}
        
        with self._lock:
            for i, key in enumerate(self.config.columns):
                self._data_buffers[key].append(all_data[i])
                sample[key] = all_data[i]
        
        # Invoke callbacks (even during burn-in for preprocessing pipeline warmup)
        for callback in self._sample_callbacks:
            try:
                callback(sample)
            except Exception as e:
                print(f"Warning: Sample callback raised exception: {e}")
        
        return sample
    
    def _check_burn_in_complete(self) -> bool:
        """
        Check if burn-in period has completed and handle transition.
        
        Returns
        -------
        bool
            True if burn-in just completed (transition), False otherwise.
        """
        if self._burn_in_complete or self.config.burn_in_seconds <= 0:
            return False
        
        elapsed = time.time() - self._start_time
        if elapsed >= self.config.burn_in_seconds:
            self._burn_in_complete = True
            print(f"\n[Burn-in Complete] {self.config.burn_in_seconds}s burn-in period finished. Resetting buffers and plots...")
            
            # Clear all data buffers - discard burn-in data
            self.clear_buffers()
            
            # Reset the start time so duration tracking starts fresh
            self._start_time = time.time()
            
            # Reset plots
            self.reset_plots()
            
            # Invoke burn-in complete callbacks (e.g., reset window buffers)
            for callback in self._burn_in_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"Warning: Burn-in callback raised exception: {e}")
            
            print("[Burn-in Complete] Data collection now active.\n")
            return True
        
        return False
    
    def _collection_loop(self) -> None:
        """Main data collection loop."""
        while self._running:
            sample = self._collect_sample()
            if sample is None:
                continue
            
            # Check if burn-in period completed
            self._check_burn_in_complete()
            
            # Check duration limit (only after burn-in complete)
            if self._burn_in_complete or self.config.burn_in_seconds <= 0:
                if self.config.save_duration_seconds > 0:
                    elapsed = time.time() - self._start_time
                    if elapsed >= self.config.save_duration_seconds:
                        self._running = False
                        break
    
    def _collection_loop_with_plot(self) -> None:
        """Data collection loop with real-time plotting (must run in main thread)."""
        while self._running:
            sample = self._collect_sample()
            if sample is None:
                continue
            
            # Check if burn-in period completed
            self._check_burn_in_complete()
            
            self._update_plot()
            
            # Check duration limit (only after burn-in complete)
            if self._burn_in_complete or self.config.burn_in_seconds <= 0:
                if self.config.save_duration_seconds > 0:
                    elapsed = time.time() - self._start_time
                    if elapsed >= self.config.save_duration_seconds:
                        self._running = False
                        break
    
    def start(self) -> 'EEGStream':
        """
        Start data collection.
        
        If use_background_thread is True, returns immediately and collection
        runs in background. Otherwise, blocks until duration is complete.
        
        Returns
        -------
        EEGStream
            Self for method chaining.
        """
        if self._inlet is None:
            self.connect()
        
        self._running = True
        self._start_time = None
        self._burn_in_complete = self.config.burn_in_seconds <= 0  # Skip burn-in if duration is 0
        
        # Print burn-in info if enabled
        if self.config.burn_in_seconds > 0:
            print(f"[Burn-in] Starting {self.config.burn_in_seconds}s burn-in period. Data will be processed but discarded...")
        
        if self.config.use_background_thread:
            # Background thread mode - graphing disabled (matplotlib not thread-safe)
            if self.config.enable_graphing:
                print("Warning: Graphing disabled in background thread mode (matplotlib is not thread-safe)")
            self._thread = threading.Thread(target=self._collection_loop, daemon=True)
            self._thread.start()
        else:
            # Foreground mode - can do graphing
            if self.config.enable_graphing:
                self._setup_plot()
            self._collection_loop_with_plot()
            
            # Save CSV if enabled
            if self.config.enable_csv_save:
                self.save_csv()
            
            # Keep plot open
            if self.config.enable_graphing:
                print("Data collection finished. Plot window will remain open.")
                plt.show()
        
        return self
    
    def stop(self) -> 'EEGStream':
        """
        Stop data collection.
        
        Returns
        -------
        EEGStream
            Self for method chaining.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self
    
    def is_running(self) -> bool:
        """Check if data collection is currently running."""
        return self._running
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get collected data as a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all collected samples.
        """
        with self._lock:
            return pd.DataFrame({k: list(v) for k, v in self._data_buffers.items()})
    
    def get_latest_samples(self, n: int = 1) -> pd.DataFrame:
        """
        Get the N most recent samples.
        
        Parameters
        ----------
        n : int
            Number of samples to retrieve.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with the N most recent samples.
        """
        df = self.get_dataframe()
        return df.tail(n)
    
    def get_buffer(self, column: str) -> List[float]:
        """
        Get the current buffer contents for a specific column.
        
        Parameters
        ----------
        column : str
            Column name (e.g., 'FZ', 'C3', 'Time').
            
        Returns
        -------
        list
            List of values in the buffer.
        """
        with self._lock:
            return list(self._data_buffers.get(column, []))
    
    def save_csv(self, path: Optional[str] = None) -> Path:
        """
        Save collected data to CSV file.
        
        Parameters
        ----------
        path : str, optional
            Output file path. Uses config.csv_path if not provided.
            
        Returns
        -------
        Path
            Path to the saved file.
        """
        save_path = Path(path or self.config.csv_path)
        df = self.get_dataframe()
        df.to_csv(save_path, index=False)
        print(f"Saved EEG data to {save_path}")
        return save_path
    
    def clear_buffers(self) -> None:
        """Clear all data buffers."""
        with self._lock:
            for buffer in self._data_buffers.values():
                buffer.clear()
        self._start_time = None
    
    '''
    @property
    def elapsed_time(self) -> float:
        """Get elapsed collection time in seconds."""
        if self._start_time is None:
            return 0.0
        with self._lock:
            times = self._data_buffers.get('Time', deque())
            if times:
                return times[-1] - self._start_time
        return 0.0
    '''
    
    @property
    def sample_count(self) -> int:
        """Get the number of samples currently in buffer."""
        with self._lock:
            times = self._data_buffers.get('Time', deque())
            return len(times)
    
    def __enter__(self) -> 'EEGStream':
        """Context manager entry."""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
        if self.config.enable_csv_save and self.sample_count > 0:
            self.save_csv()


# Convenience function for quick usage
def collect_eeg(
    duration: float = 60.0,
    stream_name: str = 'UN-2024.06.42',
    enable_graphing: bool = True,
    csv_path: Optional[str] = 'EEGdata.csv'
) -> pd.DataFrame:
    """
    Convenience function to collect EEG data with sensible defaults.
    
    Parameters
    ----------
    duration : float
        Collection duration in seconds.
    stream_name : str
        LSL stream name to connect to.
    enable_graphing : bool
        Whether to show real-time plot.
    csv_path : str, optional
        Path to save CSV. Set to None to skip saving.
        
    Returns
    -------
    pd.DataFrame
        Collected EEG data.
    """
    config = EEGStreamConfig(
        stream_name=stream_name,
        save_duration_seconds=duration,
        enable_graphing=enable_graphing,
        enable_csv_save=csv_path is not None,
        csv_path=csv_path or 'EEGdata.csv'
    )
    stream = EEGStream(config)
    stream.start()
    return stream.get_dataframe()


# Example usage when run directly
if __name__ == '__main__':
    # Example 1: Basic usage (same behavior as original script)
    print("Starting EEG collection with graphing...")
    stream = EEGStream()
    stream.start()
    
    # Example 2: Background collection with processing callback
    # def process_sample(sample):
    #     print(f"Processing sample at t={sample['Time']:.3f}")
    #
    # config = EEGStreamConfig(
    #     use_background_thread=True,
    #     enable_graphing=False,
    #     save_duration_seconds=30
    # )
    # stream = EEGStream(config)
    # stream.on_sample(process_sample)
    # stream.start()
    # 
    # # Do other work while collecting
    # while stream.is_running():
    #     print(f"Collected {stream.sample_count} samples, elapsed: {stream.elapsed_time:.1f}s")
    #     time.sleep(1)
    #
    # stream.save_csv('my_data.csv')
    # df = stream.get_dataframe()