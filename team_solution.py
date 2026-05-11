# Team members: Alexander Iodice, Kevin Chehab, Joseph Fichman, and Massimo Monaco
# Student IDs: 2533337, 2533862, 2592257, 2578851
# Required variants from ID digits:
# 2533337 -> Variant C
# 2533862 -> Variant A
# 2592257 -> Variant C
# 2578851 -> Variant A

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

BACKGROUND_COUNT = 10
ROLLING_WINDOW = 3
ANOMALY_THRESHOLD = 45
LATE_JUMP_THRESHOLD = 5


def load_csv(path):
    return pd.read_csv(path)


def prepare_decay_data(path):
    df = load_csv(path).copy()

    if "Time_s" not in df.columns or "Counts" not in df.columns:
        raise ValueError(f"{path.name} must contain Time_s and Counts columns.")

    df["Time_s"] = pd.to_numeric(df["Time_s"], errors="coerce")
    df["Counts"] = pd.to_numeric(df["Counts"], errors="coerce")
    df = df.dropna(subset=["Time_s", "Counts"])
    df = df.sort_values("Time_s").reset_index(drop=True)
    df["Trial"] = path.stem

    return df


def estimate_half_life(df, count_column="Counts"):
    first_count = df[count_column].iloc[0]
    half_count = first_count / 2

    half_rows = df[df[count_column] <= half_count]

    if half_rows.empty:
        return np.nan

    return float(half_rows["Time_s"].iloc[0])


def percent_drop(df, count_column="Counts"):
    first_count = df[count_column].iloc[0]
    last_count = df[count_column].iloc[-1]

    if first_count == 0:
        return np.nan

    return ((first_count - last_count) / first_count) * 100


def add_background_correction(df, background_count=BACKGROUND_COUNT):
    corrected = df.copy()
    corrected["Corrected_Counts"] = corrected["Counts"] - background_count
    corrected["Corrected_Counts"] = corrected["Corrected_Counts"].clip(lower=0)
    return corrected


def add_smoothing(df, window=ROLLING_WINDOW):
    smoothed = df.copy()
    smoothed["Smoothed_Counts"] = smoothed["Counts"].rolling(window=window, min_periods=1).mean()
    smoothed["Residual"] = smoothed["Counts"] - smoothed["Smoothed_Counts"]
    return smoothed


def detect_anomalies(df, threshold=ANOMALY_THRESHOLD):
    checked = df.copy()
    checked["Count_Change"] = checked["Counts"].diff()
    checked["Absolute_Change"] = checked["Count_Change"].abs()
    checked["Anomaly"] = checked["Absolute_Change"] > threshold
    return checked


def late_time_anomalies(df, threshold=LATE_JUMP_THRESHOLD):
    checked = detect_anomalies(df, ANOMALY_THRESHOLD)
    midpoint = checked["Time_s"].median()
    return checked[(checked["Time_s"] > midpoint) & (checked["Count_Change"] > threshold)]


def stability_score(df):
    return float(df["Counts"].diff().abs().mean())


def summarize_trial(df):
    corrected = add_background_correction(df)
    checked = detect_anomalies(df)

    return {
        "Trial": df["Trial"].iloc[0],
        "Initial_Count": round(df["Counts"].iloc[0], 2),
        "Minimum_Count": round(df["Counts"].min(), 2),
        "Mean_Count": round(df["Counts"].mean(), 2),
        "Estimated_Half_Life_s": estimate_half_life(df),
        "Corrected_Half_Life_s": estimate_half_life(corrected, "Corrected_Counts"),
        "Percent_Drop": round(percent_drop(df), 2),
        "Stability_Score": round(stability_score(df), 2),
        "Anomaly_Count": int(checked["Anomaly"].sum())
    }


def plot_raw_decay(df):
    plt.figure()
    plt.plot(df["Time_s"], df["Counts"], marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.title("Radiation Decay: Counts vs Time")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "raw_decay_plot.png", bbox_inches="tight")
    plt.close()


def plot_half_life_comparison(summary_df):
    ranked = summary_df.sort_values("Estimated_Half_Life_s")

    plt.figure()
    plt.bar(ranked["Trial"], ranked["Estimated_Half_Life_s"])
    plt.xlabel("Trial")
    plt.ylabel("Estimated Half-Life (s)")
    plt.title("Half-Life Comparison")
    plt.xticks(rotation=25)
    plt.savefig(OUTPUT_DIR / "half_life_comparison.png", bbox_inches="tight")
    plt.close()


def plot_corrected_half_life_comparison(summary_df):
    x = np.arange(len(summary_df))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, summary_df["Estimated_Half_Life_s"], width, label="Raw")
    plt.bar(x + width / 2, summary_df["Corrected_Half_Life_s"], width, label="Corrected")
    plt.xticks(x, summary_df["Trial"], rotation=25)
    plt.xlabel("Trial")
    plt.ylabel("Half-Life (s)")
    plt.title("Raw vs Corrected Half-Life")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "corrected_half_life_comparison.png", bbox_inches="tight")
    plt.close()


def plot_smoothed_vs_raw(df):
    smoothed = add_smoothing(df)

    plt.figure()
    plt.plot(smoothed["Time_s"], smoothed["Counts"], marker="o", label="Raw Counts")
    plt.plot(smoothed["Time_s"], smoothed["Smoothed_Counts"], marker="o", label="Smoothed Counts")
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.title("Raw Counts vs Rolling Average")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "smoothed_vs_raw_curve.png", bbox_inches="tight")
    plt.close()


def plot_stability(summary_df):
    ranked = summary_df.sort_values("Stability_Score")

    plt.figure()
    plt.bar(ranked["Trial"], ranked["Stability_Score"])
    plt.xlabel("Trial")
    plt.ylabel("Average Absolute Count Change")
    plt.title("Trial Stability Comparison")
    plt.xticks(rotation=25)
    plt.savefig(OUTPUT_DIR / "stability_comparison.png", bbox_inches="tight")
    plt.close()


def plot_dashboard(combined_df, summary_df):
    first_trial = combined_df["Trial"].iloc[0]
    one = combined_df[combined_df["Trial"] == first_trial]
    smoothed_one = add_smoothing(one)

    plt.figure(figsize=(11, 8))

    plt.subplot(2, 2, 1)
    for trial, df in combined_df.groupby("Trial"):
        plt.plot(df["Time_s"], df["Counts"], marker="o", label=trial)
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.title("Counts vs Time")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.bar(summary_df["Trial"], summary_df["Estimated_Half_Life_s"])
    plt.title("Estimated Half-Life")
    plt.xticks(rotation=25)

    plt.subplot(2, 2, 3)
    plt.plot(smoothed_one["Time_s"], smoothed_one["Counts"], marker="o", label="Raw")
    plt.plot(smoothed_one["Time_s"], smoothed_one["Smoothed_Counts"], marker="o", label="Smoothed")
    plt.title("Smoothed vs Raw Curve")
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.bar(summary_df["Trial"], summary_df["Stability_Score"])
    plt.title("Dashboard Stability Panel")
    plt.xticks(rotation=25)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decay_dashboard.png", bbox_inches="tight")
    plt.close()


def phase1(files):
    first_file = files[0]
    df = prepare_decay_data(first_file)

    summary_df = pd.DataFrame([summarize_trial(df)])
    summary_df.to_csv(OUTPUT_DIR / "phase1_summary.csv", index=False)

    plot_raw_decay(df)

    (OUTPUT_DIR / "phase1_reflection.md").write_text(
        "# Phase 1 Reflection\n\n"
        "The Phase 1 code loads a decay CSV file, validates the Time_s and Counts columns, "
        "sorts the readings by time, creates a raw decay plot, and exports a summary table. "
        "The half-life is estimated with the first-half method by finding the first time where "
        "the count value falls to half of the initial count or lower.\n",
        encoding="utf-8"
    )


def phase2(files):
    summaries = []
    anomaly_frames = []

    for file in files:
        df = prepare_decay_data(file)
        corrected = add_background_correction(df)
        checked = detect_anomalies(df)

        summaries.append(summarize_trial(df))

        anomalies = checked[checked["Anomaly"]].copy()
        if not anomalies.empty:
            anomaly_frames.append(anomalies)

    summary_df = pd.DataFrame(summaries).sort_values("Estimated_Half_Life_s")
    summary_df["Half_Life_Rank"] = range(1, len(summary_df) + 1)
    summary_df.to_csv(OUTPUT_DIR / "phase2_summary.csv", index=False)

    if anomaly_frames:
        pd.concat(anomaly_frames, ignore_index=True).to_csv(OUTPUT_DIR / "phase2_anomalies.csv", index=False)
    else:
        pd.DataFrame(columns=["Trial", "Time_s", "Counts", "Count_Change", "Absolute_Change", "Anomaly"]).to_csv(
            OUTPUT_DIR / "phase2_anomalies.csv", index=False
        )

    plot_half_life_comparison(summary_df)

    (OUTPUT_DIR / "phase2_notes.md").write_text(
        "# Phase 2 Notes\n\n"
        "Phase 2 compares at least three decay trials. For each trial, the code estimates half-life, "
        "calculates total percent drop, ranks the trials by half-life, and creates a comparison bar chart.\n\n"
        "Student-ID modules included:\n\n"
        "- Variant A: background-count correction before computing corrected half-life.\n"
        "- Variant C: anomaly detection using unusually large count differences between readings.\n",
        encoding="utf-8"
    )


def phase3(files):
    detailed = []
    summaries = []
    late_anomaly_frames = []

    for file in files:
        df = prepare_decay_data(file)
        df = add_background_correction(df)
        df = add_smoothing(df)
        df = detect_anomalies(df)

        detailed.append(df)
        summaries.append(summarize_trial(df))

        late = late_time_anomalies(df)
        if not late.empty:
            late_anomaly_frames.append(late)

    combined_df = pd.concat(detailed, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values("Stability_Score")

    combined_df.to_csv(OUTPUT_DIR / "combined_decay_data.csv", index=False)

    if late_anomaly_frames:
        pd.concat(late_anomaly_frames, ignore_index=True).to_csv(OUTPUT_DIR / "late_time_anomaly_report.csv", index=False)
    else:
        pd.DataFrame(columns=["Trial", "Time_s", "Counts", "Count_Change", "Absolute_Change", "Anomaly"]).to_csv(
            OUTPUT_DIR / "late_time_anomaly_report.csv", index=False
        )

    plot_corrected_half_life_comparison(summary_df)
    plot_smoothed_vs_raw(combined_df[combined_df["Trial"] == combined_df["Trial"].iloc[0]])
    plot_dashboard(combined_df, summary_df)
    plot_stability(summary_df)

    most_stable = summary_df.iloc[0]
    least_stable = summary_df.iloc[-1]

    report = (
        "Phase 3 Final Report\n"
        "====================\n\n"
        f"Most stable trial: {most_stable['Trial']}\n"
        f"Least stable trial: {least_stable['Trial']}\n\n"
        "Stability was estimated using the average absolute difference between consecutive count values. "
        "A smaller stability score means that the trial had smoother count changes between readings.\n\n"
        "Variant A was included by comparing raw half-life with corrected half-life after subtracting "
        f"a background count of {BACKGROUND_COUNT}.\n\n"
        f"Variant C was included by detecting unusually large count differences and by exporting a late-time "
        f"anomaly report for irregular jumps greater than {LATE_JUMP_THRESHOLD} counts after the midpoint of the trial.\n\n"
        "The cleaned combined dataset includes raw counts, corrected counts, rolling-average smoothed counts, "
        "residuals, count changes, and anomaly flags for all trial files.\n"
    )

    (OUTPUT_DIR / "phase3_report.txt").write_text(report, encoding="utf-8")


def main():
    files = sorted(DATA_DIR.glob("*.csv"))

    if len(files) == 0:
        raise FileNotFoundError("No CSV files were found in the data folder.")

    phase1(files)
    phase2(files)
    phase3(files)

    print("Radiation decay analysis complete.")
    print(f"Processed {len(files)} file(s).")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()



# 1. Count values above a threshold
def count_above_threshold(counts, threshold):
    total = 0

    for value in counts:
        if value > threshold:
            total += 1

    return total


# 2. Count values below a threshold
def count_below_threshold(counts, threshold):
    total = 0

    for value in counts:
        if value < threshold:
            total += 1

    return total


# 3. Count values between two numbers
def count_between(counts, low, high):
    total = 0

    for value in counts:
        if low <= value <= high:
            total += 1

    return total


# 4. Find the first value below a threshold
def first_below_threshold(counts, threshold):
    for value in counts:
        if value < threshold:
            return value

    return None


# 5. Count how many values are below half the initial count
def count_below_half_initial(counts):
    initial_count = counts.iloc[0]
    half_count = initial_count / 2
    total = 0

    for value in counts:
        if value < half_count:
            total += 1

    return total


# 6. Find first time count goes below half initial count
def find_half_life_time(df):
    initial_count = df["Counts"].iloc[0]
    half_count = initial_count / 2

    for i in range(len(df)):
        if df["Counts"].iloc[i] <= half_count:
            return df["Time_s"].iloc[i]

    return None


# 7. Find the maximum count using a loop
def find_max_count(counts):
    maximum = counts.iloc[0]

    for value in counts:
        if value > maximum:
            maximum = value

    return maximum


# 8. Find the minimum count using a loop
def find_min_count(counts):
    minimum = counts.iloc[0]

    for value in counts:
        if value < minimum:
            minimum = value

    return minimum


# ============================================================
# PHASE 2 — ADDING COLUMNS TO SUMMARY DATAFRAME
# ============================================================

# Use these after you already have summary_df created.

# 1. Add count range
# summary_df["Count_Range"] = summary_df["Max_Count"] - summary_df["Min_Count"]

# 2. Add half of the initial count
# summary_df["Half_Initial_Count"] = summary_df["Initial_Count"] / 2

# 3. Add count drop
# summary_df["Count_Drop"] = summary_df["Initial_Count"] - summary_df["Final_Count"]

# 4. Add percent drop
# summary_df["Percent_Drop"] = ((summary_df["Initial_Count"] - summary_df["Final_Count"]) / summary_df["Initial_Count"]) * 100

# 5. Add average count per second
# summary_df["Average_Count_Per_Second"] = summary_df["Average_Count"] / summary_df["Total_Time_s"]

# 6. Add fast decay column
# summary_df["Fast_Decay"] = summary_df["Half_Life_s"] < 50

# 7. Add difference from average half-life
# overall_avg_half_life = summary_df["Half_Life_s"].mean()
# summary_df["Difference_From_Average"] = summary_df["Half_Life_s"] - overall_avg_half_life

# 8. Add difference from average count
# overall_avg_count = summary_df["Average_Count"].mean()
# summary_df["Difference_From_Avg_Count"] = summary_df["Average_Count"] - overall_avg_count


# ============================================================
# PHASE 3 — FILTERING DATAFRAMES
# ============================================================

# Use these after you already have df created.

# 1. Filter rows after 40 seconds and below average count
# trial_avg = df["Counts"].mean()
# filtered = df[(df["Time_s"] >= 40) & (df["Counts"] < trial_avg)]
# print(filtered)

# 2. Filter rows where counts are above average
# trial_avg = df["Counts"].mean()
# filtered = df[df["Counts"] > trial_avg]
# print(filtered)

# 3. Filter rows where time is between 20 and 60 seconds
# filtered = df[(df["Time_s"] >= 20) & (df["Time_s"] <= 60)]
# print(filtered)

# 4. Filter rows where counts are below half the initial count
# initial_count = df["Counts"].iloc[0]
# half_count = initial_count / 2
# filtered = df[df["Counts"] <= half_count]
# print(filtered)

# 5. Filter rows after 30 seconds and counts above 100
# filtered = df[(df["Time_s"] >= 30) & (df["Counts"] > 100)]
# print(filtered)

# 6. Filter suspicious data: counts less than 0 or missing
# filtered = df[(df["Counts"] < 0) | (df["Counts"].isna())]
# print(filtered)

# 7. Filter rows where counts are between 50 and 150
# filtered = df[(df["Counts"] >= 50) & (df["Counts"] <= 150)]
# print(filtered)

# 8. Filter rows before 60 seconds and above average
# trial_avg = df["Counts"].mean()
# filtered = df[(df["Time_s"] < 60) & (df["Counts"] > trial_avg)]
# print(filtered)



