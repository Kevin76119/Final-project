import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# FILE SETUP (SUPER SIMPLE)
# =========================
FILE = Path("decay_c.csv")
OUTPUT_DIR = Path("outputs")

try:
    OUTPUT_DIR.mkdir(exist_ok=True)
except:
    OUTPUT_DIR = Path.cwd()
# =========================
# FUNCTIONS
# =========================

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()
    df.columns = [col.lower() for col in df.columns]
    return df.sort_values(df.columns[0])

def estimate_half_life(df):
    time_col = df.columns[0]
    count_col = df.columns[1]

    initial = df[count_col].iloc[0]
    half = initial / 2

    for i in range(len(df)):
        if df[count_col].iloc[i] <= half:
            return df[time_col].iloc[i]
    return None

def find_irregular_points(df):
    count_col = df.columns[1]
    df = df.copy()
    df["diff"] = df[count_col].diff().abs()

    threshold = df["diff"].mean() * 2
    return df[df["diff"] > threshold]

def late_time_anomalies(df):
    cutoff = int(len(df) * 0.7)
    return find_irregular_points(df.iloc[cutoff:])

# =========================
# MAIN
# =========================

def main():
    # Check file exists
    if not FILE.exists():
        print("❌ File not found: decay_c.csv")
        return

    print("✅ File found, processing...")

    df = load_data(FILE)
    df = clean_data(df)

    time_col = df.columns[0]
    count_col = df.columns[1]

    # Basic stats
    initial = df[count_col].iloc[0]
    minimum = df[count_col].min()
    mean = df[count_col].mean()

    # Half-life
    hl = estimate_half_life(df)

    # Percent drop
    percent_drop = ((initial - df[count_col].iloc[-1]) / initial) * 100

    # Anomalies
    irregular = find_irregular_points(df)
    late_anom = late_time_anomalies(df)

    # =========================
    # PRINT RESULTS
    # =========================
    print("\n--- RESULTS ---")
    print(f"Initial count: {initial}")
    print(f"Minimum count: {minimum}")
    print(f"Mean count: {mean:.2f}")
    print(f"Estimated half-life: {hl}")
    print(f"Percent drop: {percent_drop:.2f}%")
    print(f"Irregular points: {len(irregular)}")
    print(f"Late anomalies: {len(late_anom)}")

    # =========================
    # SAVE CSV
    # =========================
    summary = pd.DataFrame([{
        "half_life": hl,
        "percent_drop": percent_drop,
        "num_irregular": len(irregular),
        "late_anomalies": len(late_anom)
    }])

    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    # =========================
    # PLOT
    # =========================
    plt.figure()
    plt.plot(df[time_col], df[count_col], label="Counts")
    plt.scatter(irregular[time_col], irregular[count_col], label="Anomalies")
    plt.legend()
    plt.title("Decay Analysis")

    plt.savefig(OUTPUT_DIR / "plot.png")
    plt.close()

    # =========================
    # REPORT
    # =========================
    with open(OUTPUT_DIR / "report.txt", "w") as f:
        f.write("Radiation Decay Report\n\n")
        f.write(f"Half-life: {hl}\n")
        f.write(f"Percent drop: {percent_drop:.2f}%\n")
        f.write(f"Irregular points: {len(irregular)}\n")
        f.write(f"Late anomalies: {len(late_anom)}\n")

    print("\n✅ DONE! Check the outputs folder.")

# =========================

if __name__ == "__main__":
    main()
