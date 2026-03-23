import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from app.config_paths import DATA_DIR, REPORTS_DIR

# -----------------------------
# PARTICIPANT CONFIG
# -----------------------------
def plot_data(participant_id: str):
    csv_file = os.path.join(DATA_DIR, f"auditory_test_{participant_id}.csv")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"{csv_file} not found")
    
    # Load data
    df_trials = pd.read_csv(csv_file)
    
    # -----------------------------
    # CURRENT SESSION DATA
    # -----------------------------
    last_session_date = df_trials['session_date'].max()
    df_session = df_trials[df_trials['session_date'] == last_session_date]
    
    # Compute stats
    total_trials = len(df_session)
    total_correct = df_session['Correct'].sum()
    accuracy = round(total_correct / total_trials * 100, 2)
    average_threshold = df_session['Frequency_Diff'].mean()
    
    # -----------------------------
    # SAFE TIMESTAMP FOR FILENAMES
    # -----------------------------
    safe_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # -----------------------------
    # PLOT 1 — CURRENT SESSION
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.scatter(
        df_session.loc[df_session['Correct'] == 0, 'Trial'],
        df_session.loc[df_session['Correct'] == 0, 'Frequency_Diff'],
        color='red', edgecolors='black', s=100, label='Incorrect', zorder=5
    )
    plt.scatter(
        df_session.loc[df_session['Correct'] == 1, 'Trial'],
        df_session.loc[df_session['Correct'] == 1, 'Frequency_Diff'],
        color='green', edgecolors='black', s=100, label='Correct', zorder=5
    )
    plt.plot(df_session['Trial'], df_session['Frequency_Diff'],
             color='blue', linewidth=1.5, label='Frequency Difference')
    plt.xticks(range(1, df_session['Trial'].max() + 1))
    plt.xlabel('Trial')
    plt.ylabel('Frequency Difference (Hz)')
    plt.title(f'Participant {participant_id} - Current Session Performance (Accuracy: {accuracy}%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    session_plot_path = os.path.join(REPORTS_DIR,
                                     f"{participant_id}_performance_plot-{safe_timestamp}.png")
    plt.savefig(session_plot_path)
    plt.show()
    
    print("\n=== CURRENT SESSION SUMMARY ===")
    print(f"Total Trials: {total_trials}")
    print(f"Correct Responses: {total_correct}")
    print(f"Accuracy: {accuracy}%")
    print(f"Average Threshold: {average_threshold:.2f} Hz")
    print(f"Session plot saved to: {session_plot_path}")
    
    # -----------------------------
    # PLOT 2 — HISTORICAL OVERALL
    # -----------------------------
    # new
    df_trials['session_date'] = pd.to_datetime(df_trials['session_date'], format='mixed', errors='raise')
    df_trials = df_trials.sort_values('session_date')
    df_trials['Session_Num'] = pd.factorize(df_trials['session_date'])[0] + 1
    
    session_summary = df_trials.groupby('Session_Num').agg(
        Accuracy=('Correct', 'mean'),
        Avg_Freq_Diff=('Frequency_Diff', 'mean')
    ).reset_index()
    session_summary['Accuracy'] *= 100
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(session_summary['Session_Num'], session_summary['Accuracy'],
            color='skyblue', edgecolor='black', label='Accuracy (%)')
    ax1.set_xlabel('Session Number')
    ax1.set_ylabel('Accuracy (%)', color='blue')
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Accuracy trend line
    z = np.polyfit(session_summary['Session_Num'], session_summary['Accuracy'], deg=1)
    p = np.poly1d(z)
    ax1.plot(session_summary['Session_Num'], p(session_summary['Session_Num']),
             color='blue', linestyle='--', label='Accuracy Trend')
    
    # Secondary axis for average frequency difference
    ax2 = ax1.twinx()
    ax2.plot(session_summary['Session_Num'], session_summary['Avg_Freq_Diff'],
             color='red', marker='o', label='Avg Frequency Diff (Hz)')
    ax2.set_ylabel('Avg Frequency Diff (Hz)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.title('Participant Session Performance Overview')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    overall_plot_path = os.path.join(REPORTS_DIR,
                                     f"{participant_id}_overall_performance_plot-{safe_timestamp}.png")
    plt.savefig(overall_plot_path)
    plt.show()
    
    print(f"Overall performance plot saved to: {overall_plot_path}")