# Import required libraries
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from scipy import signal
from obspy.signal.trigger import classic_sta_lta, trigger_onset

# # 1. Read in Apollo 12 Grade A catalog
cat_directory = './data/lunar/training/catalogs/'
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
cat = pd.read_csv(cat_file)
print(cat.head())

# 2. Select a detection: Choose the first event in the catalog
row = cat.iloc[6]
arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')
arrival_time_rel = row['time_rel(sec)']
test_filename = row.filename
print(f"Selected file: {test_filename}")
print(f"Absolute time of arrival: {arrival_time}")
print(f"Relative time of arrival: {arrival_time_rel} seconds")

# 3. Read and plot the corresponding CSV data
data_directory = './data/lunar/training/data/S12_GradeA/'
csv_file = f'{data_directory}{test_filename}.csv'
data_cat = pd.read_csv(csv_file)

# # FIRST METHOD FOLLOWS
#
# # Extract time and velocity
# csv_times = np.array(data_cat['time_rel(sec)'].tolist())
# csv_data = np.array(data_cat['velocity(m/s)'].tolist())
#
# # Plot the trace
# fig, ax = plt.subplots(1, 1, figsize=(10, 3))
# ax.plot(csv_times, csv_data)
#
# # Make the plot pretty
# ax.set_xlim([min(csv_times), max(csv_times)])
# ax.set_ylabel('Velocity (m/s)')
# ax.set_xlabel('Time (s)')
# ax.set_title(f'{test_filename}', fontweight='bold')
#
# # Plot where the arrival time is
# arrival_line = ax.axvline(x=arrival_time_rel, c='red', label='Rel. Arrival')
# ax.legend(handles=[arrival_line])
# plt.show()
#
# SECOND METHOD: 4. Alternatively, read the miniSEED file corresponding to that
# detection

mseed_file = f'{data_directory}{test_filename}.mseed'
st = read(mseed_file)
print(st)

# Get the data from the stream
tr = st[0].copy()
tr_times = tr.times()
tr_data = tr.data

# Calculate relative arrival time using start time
starttime = tr.stats.starttime.datetime
arrival = (arrival_time - starttime).total_seconds()

# Plot the miniSEED trace
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(tr_times, tr_data)
ax.axvline(x=arrival, color='red', label='Rel. Arrival')
ax.legend(loc='upper left')
ax.set_xlim([min(tr_times), max(tr_times)])
ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')
ax.set_title(f'{test_filename}', fontweight='bold')
plt.show()

# 5. Apply bandpass filter between 0.5 Hz and 1.0 Hz
minfreq = 0.5
maxfreq = 1.0
st_filt = st.copy()
st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
tr_filt = st_filt[0]
tr_times_filt = tr_filt.times()
tr_data_filt = tr_filt.data

# Plot filtered trace
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(tr_times_filt, tr_data_filt)
ax.axvline(x=arrival, color='red', label='Detection')
ax.legend(loc='upper left')
ax.set_xlim([min(tr_times_filt), max(tr_times_filt)])
ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')
ax.set_title(f'{test_filename} (Filtered)', fontweight='bold')
plt.show()

# 6. Plot spectrogram of filtered data
f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)
fig, ax2 = plt.subplots(1, 1, figsize=(10, 5))
vals = ax2.pcolormesh(t, f, sxx, cmap='jet', vmax=5e-17)
ax2.set_xlim([min(tr_times_filt), max(tr_times_filt)])
ax2.set_xlabel(f'Time (s)', fontweight='bold')
ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
ax2.axvline(x=arrival, c='red')
plt.colorbar(vals, orientation='horizontal').set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')
plt.show()

# 7. Apply STA/LTA detection algorithm
df = tr.stats.sampling_rate
sta_len = 120  # Short-term window length (seconds)
lta_len = 600  # Long-term window length (seconds)

# Calculate STA/LTA characteristic function
cft = classic_sta_lta(tr_data, int(sta_len * df), int(lta_len * df))

# Plot STA/LTA characteristic function
fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.plot(tr_times, cft)
ax.set_xlim([min(tr_times), max(tr_times)])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Characteristic function')
plt.show()

# Define trigger values and calculate on/off times
thr_on = 4
thr_off = 1.5
on_off = np.array(trigger_onset(cft, thr_on, thr_off))

# Plot detection on/off times
fig, ax = plt.subplots(1, 1, figsize=(12, 3))
for triggers in on_off:
    ax.axvline(x=tr_times[triggers[0]], color='red', label='Trig. On')
    ax.axvline(x=tr_times[triggers[1]], color='purple', label='Trig. Off')

ax.plot(tr_times, tr_data)
ax.set_xlim([min(tr_times), max(tr_times)])
ax.legend()
plt.show()

# 8. Export detection results to a catalog
detection_times = []
fnames = []
for triggers in on_off:
    on_time = starttime + timedelta(seconds=tr_times[triggers[0]])
    on_time_str = datetime.strftime(on_time, '%Y-%m-%dT%H:%M:%S.%f')
    detection_times.append(on_time_str)
    fnames.append(test_filename)

detect_df = pd.DataFrame(data={
    'filename': fnames,
    'time_abs(%Y-%m-%dT%H:%M:%S.%f)': detection_times,
    'time_rel(sec)': [tr_times[triggers[0]] for triggers in on_off]
})

# Output detection results to CSV
output_catalog_path = './output_catalog.csv'
detect_df.to_csv(output_catalog_path, index=False)
print(f'Detection catalog saved to {output_catalog_path}')

