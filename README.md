# Radiation Decay & Half-Life Analysis Suite

This project analyzes simulated radioactive decay data from CSV files. It loads and validates the data, estimates half-life, compares trials, creates plots, and exports summary files for reporting.

## Team Student IDs and Variants

- 2533337: Variant C
- 2533862: Variant A
- 2592257: Variant C
- 2578851: Variant A

The final code includes Variant A and Variant C because those are the variants required by the team IDs.

## How to Run

From the project folder, run:

```bash
python team_solution.py
```

The program reads every CSV file in the `data/` folder and saves all output files in the `outputs/` folder.

## Required Python Libraries

```bash
pip install pandas numpy matplotlib
```

## Input Files

The included sample data files are:

- `data/decay_lab_A.csv`
- `data/decay_lab_B.csv`
- `data/decay_lab_C.csv`

Each file must contain:

- `Time_s`
- `Counts`

## Required Outputs

Phase 1:

- `raw_decay_plot.png`
- `phase1_summary.csv`
- `phase1_reflection.md`

Phase 2:

- `phase2_summary.csv`
- `half_life_comparison.png`
- `phase2_notes.md`
- `phase2_anomalies.csv`

Phase 3:

- `combined_decay_data.csv`
- `decay_dashboard.png`
- `phase3_report.txt`
- `corrected_half_life_comparison.png`
- `smoothed_vs_raw_curve.png`
- `stability_comparison.png`
- `late_time_anomaly_report.csv`

## Half-Life Method

The estimated half-life uses the first-half method:

1. Read the first count value.
2. Divide it by 2.
3. Find the first time where the count is less than or equal to half of the first count.

## Stability Method

The most stable trial is the one with the smallest average absolute difference between consecutive count values. A smaller value means the count readings changed more smoothly.
