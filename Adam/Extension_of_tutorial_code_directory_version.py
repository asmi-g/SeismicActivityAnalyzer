import argparse
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from scipy import signal
from obspy.signal.trigger import classic_sta_lta, trigger_onset

# Function to process CSV file
def process_csv(csv_file):
    data_cat = pd.read_csv(csv_file)

    # Extract time and velocity
    csv_times = np.array(data_cat['time_rel(sec)'].tolist())
    csv_data = np.array(data_cat['velocity(m/s)'].tolist())

    # Plot the trace
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(csv_times, csv_data)
    ax.set_xlim([min(csv_times), max(csv_times)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{csv_file}', fontweight='bold')
    plt.show()

    return csv_times, csv_data

# Function to process miniSEED file
def process_mseed(mseed_file):
    st = read(mseed_file)
    tr = st[0].copy()
    tr_times = tr.times()
    tr_data = tr.data

    # Plot the miniSEED trace
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(tr_times, tr_data)
    ax.set_xlim([min(tr_times), max(tr_times)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{mseed_file}', fontweight='bold')
    plt.show()

    return tr_times, tr_data, tr.stats.starttime.datetime

# Function to apply bandpass filter
def apply_bandpass_filter(tr_data, sampling_rate, minfreq=0.5, maxfreq=1.0):
    sos = signal.butter(4, [minfreq, maxfreq], btype='bandpass', fs=sampling_rate, output='sos')
    filtered_data = signal.sosfilt(sos, tr_data)

    return filtered_data

# Function to detect events using STA/LTA
def detect_events(tr_data, tr_times, sampling_rate, sta_len=1, lta_len=10, thr_on=4, thr_off=1.5):
    cft = classic_sta_lta(tr_data, int(sta_len * sampling_rate), int(lta_len * sampling_rate))
    on_off = np.array(trigger_onset(cft, thr_on, thr_off))

    # Plot STA/LTA and mark trigger points
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(tr_times, cft)
    for triggers in on_off:
        ax.axvline(x=tr_times[triggers[0]], color='red', label='Trig. On')
        ax.axvline(x=tr_times[triggers[1]], color='purple', label='Trig. Off')

    ax.set_xlim([min(tr_times), max(tr_times)])
    plt.show()

    return on_off

# Function to save detection results
def save_detection_results(on_off, tr_times, starttime, filename):
    detection_times = []
    fnames = []

    for triggers in on_off:
        on_time = starttime + timedelta(seconds=tr_times[triggers[0]])
        on_time_str = datetime.strftime(on_time, '%Y-%m-%dT%H:%M:%S.%f')
        detection_times.append(on_time_str)
        fnames.append(filename)

    detect_df = pd.DataFrame(data={
        'filename': fnames,
        'time_abs(%Y-%m-%dT%H:%M:%S.%f)': detection_times,
        'time_rel(sec)': [tr_times[triggers[0]] for triggers in on_off]
    })

    output_catalog_path = './output_catalog.csv'
    detect_df.to_csv(output_catalog_path, index=False)
    print(f'Detection catalog saved to {output_catalog_path}')

# Main function to process an arbitrary file
def process_seismic_file(file_path):
    file_ext = os.path.splitext(file_path)[1]

    if file_ext == '.csv':
        print(f"Processing CSV file: {file_path}")
        csv_times, csv_data = process_csv(file_path)
        filtered_data = apply_bandpass_filter(csv_data, sampling_rate=1.0)  # Assuming a sample rate of 1.0 for CSV
        on_off = detect_events(filtered_data, csv_times, sampling_rate=1.0)
        save_detection_results(on_off, csv_times, starttime=None, filename=file_path)

    elif file_ext == '.mseed':
        print(f"Processing miniSEED file: {file_path}")
        tr_times, tr_data, starttime = process_mseed(file_path)
        filtered_data = apply_bandpass_filter(tr_data, sampling_rate=100.0)  # Assuming 100 Hz for miniSEED
        on_off = detect_events(filtered_data, tr_times, sampling_rate=100.0)
        save_detection_results(on_off, tr_times, starttime=starttime, filename=file_path)

    else:
        print(f"Unsupported file format: {file_ext}")

# Main function to process all files in a directory
def process_directory(directory_path):
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter files by supported extensions
    supported_files = [f for f in files if f.endswith('.csv') or f.endswith('.mseed')]

    # Process each file
    for file in supported_files:
        file_path = os.path.join(directory_path, file)
        print(f"Processing file: {file_path}")
        process_seismic_file(file_path)

# Parse arguments from the terminal using argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process seismic data from CSV or miniSEED files in a directory.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing seismic data files (CSV or miniSEED).")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main processing function with the provided directory path
    process_directory(args.directory_path)
