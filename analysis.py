"""Assignment 03: summarize a telemetry ward's systolic readings.

Run from the assignment directory with the project environment active:

    python3 analysis.py
"""

import numpy as np


def load_readings(filename):
    """Return (patient_ids, monitors, hour_columns, readings) from the supplied CSV.

    readings is a 2D array of integers: one row per patient, one column per
    monitored hour, in the order the header lists them.
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    header = lines[0].strip().split(",")
    rows = [line.strip().split(",") for line in lines[1:] if line.strip()]

    patient_ids = np.array([row[0] for row in rows])
    monitors = np.array([row[1] for row in rows])
    hour_columns = np.array(header[2:])
    readings = np.array([row[2:] for row in rows]).astype(int)
    return patient_ids, monitors, hour_columns, readings


def main():
    patient_ids, monitors, hour_columns, readings = load_readings("data/bp_readings.csv")
    print(f"Loaded {readings.shape[0]} patients x {readings.shape[1]} hours")

    patients = len(patient_ids)
    total_readings = readings.size
    mean_sbp = readings.mean()
    sd_sbp = readings.std()
    min_sbp = readings.min()
    max_sbp = readings.max()

    patient_means = readings.mean(axis=1)
    stage2_patients = np.sum(patient_means >= 140)

    highest_patient_idx = patient_means.argmax()
    highest_patient = patient_ids[highest_patient_idx]
    highest_patient_mean = patient_means[highest_patient_idx]

    hourly_means = readings.mean(axis=0)
    peak_hour_idx = hourly_means.argmax()
    peak_hour_column = hour_columns[peak_hour_idx]
    peak_hour_mean = hourly_means[peak_hour_idx]

    unique_monitors = np.array(sorted(set(monitors)))
    monitor_averages = np.array([patient_means[monitors == m].mean() for m in unique_monitors])

    high_monitor_idx = monitor_averages.argmax()
    high_monitor = unique_monitors[high_monitor_idx]
    high_monitor_avg = monitor_averages[high_monitor_idx]

    other_monitors_mask = (monitors != high_monitor)
    other_monitors_avg = patient_means[other_monitors_mask].mean()
    monitor_offset = high_monitor_avg - other_monitors_avg
    stage2_other_monitors = np.sum(patient_means[other_monitors_mask] >= 140)

    with open("output/vitals_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"patients: {patients}\n")
        f.write(f"readings: {total_readings}\n")
        f.write(f"mean_sbp: {mean_sbp}\n")
        f.write(f"sd_sbp: {sd_sbp}\n")
        f.write(f"min_sbp: {min_sbp}\n")
        f.write(f"max_sbp: {max_sbp}\n")
        f.write(f"stage2_patients: {stage2_patients}\n")
        f.write(f"highest_patient: {highest_patient}\n")
        f.write(f"highest_patient_mean: {highest_patient_mean}\n")
        f.write(f"peak_hour_column: {peak_hour_column}\n")
        f.write(f"peak_hour_mean: {peak_hour_mean}\n")
        f.write(f"high_monitor: {high_monitor}\n")
        f.write(f"monitor_offset: {monitor_offset}\n")
        f.write(f"stage2_other_monitors: {stage2_other_monitors}\n")



    if __name__ == "__main__":
        main()
