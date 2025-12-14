from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import time

# initialize the streaming layer
streams = resolve_byprop('name', 'UN-2024.06.42')
if not streams:
   raise RuntimeError("No LSL stream found with name 'UN-2024.06.42'")
inlet = StreamInlet(streams[0])

# initialize the columns of your data and your dictionary to capture the data.
columns = [
   'Time', 'FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8',
   'AccX', 'AccY', 'AccZ', 'Gyro1', 'Gyro2', 'Gyro3', 'Battery', 'Counter', 'Validation'
]

# real-time plotting configuration
sample_rate = 250  # Hz
buffer_seconds = 10  # show last N seconds in plot
maxlen = sample_rate * buffer_seconds

# deques for time and each signal column (excluding 'Time' which is the x-axis)
data_buffers = {col: deque(maxlen=maxlen) for col in columns}

# matplotlib setup: create a grid of subplots to show all columns except 'Time'
plot_columns = [c for c in columns if c != 'Time']
n_plots = len(plot_columns)
rows = 5
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(16, 12), sharex=True)
axes = axes.flatten()

lines = {}
for i, col in enumerate(plot_columns):
   ax = axes[i]
   lines[col], = ax.plot([], [], lw=1)
   ax.set_title(col)
   ax.grid(True, alpha=0.3)

# hide any unused axes if grid > number of plots
for j in range(n_plots, rows * cols):
   fig.delaxes(axes[j])

plt.tight_layout()

start_time = None
save_duration_seconds = 60  # keep collecting for CSV save

def update_plot():
   # update each line with current buffer contents
   t = list(data_buffers['Time'])
   if not t:
      return
   t0 = t[0]
   # normalize time to start of buffer for nicer x-axis
   t_rel = [ti - t0 for ti in t]
   for col in plot_columns:
      y = list(data_buffers[col])
      lines[col].set_data(t_rel, y)
      ax = lines[col].axes
      ax.relim()
      ax.autoscale_view()
      ax.set_xlim(left=0, right=max(t_rel) if t_rel else buffer_seconds)
   plt.pause(0.001)

# data collection + live update loop
finished = False
while not finished:
   data, timestamp = inlet.pull_sample()
   if start_time is None:
      start_time = timestamp

   # concatenate timestamp and data
   all_data = [timestamp] + data

   # append to buffers
   for i, key in enumerate(columns):
      data_buffers[key].append(all_data[i])

   # refresh plots
   update_plot()

   # stop after desired duration for saving
   if (timestamp - start_time) >= save_duration_seconds:
      finished = True

# save collected data to CSV
data_df = pd.DataFrame({k: list(v) for k, v in data_buffers.items()})
data_df.to_csv('EEGdata.csv', index=False)
print("Saved EEG data to EEGdata.csv")

# keep plot window open after finishing collection
print("Data collection finished. Plot window will remain open.")
plt.show()