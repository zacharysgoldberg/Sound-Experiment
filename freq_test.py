import os
import numpy as np
from numpy import random
import simpleaudio as sa
import pandas as pd
from matplotlib import pyplot as plt
import time
from datetime import datetime
# -----------------------------
# CONFIGURATION
# -----------------------------
config = {
    "base_freq": 600,              # Base frequency in Hz
    "initial_diff": 5,             # Initial frequency difference
    "tone_duration": 0.5,          # Duration of each tone in seconds
    "step_diff": 0.2,              # Step size for staircase adjustment
    "n_reverse": 3,                # Number of allowed incorrect trials
    "silence_between_tones": 1.0,  # Seconds of silence between tones
    "max_volume": 16000,           # Max amplitude for tones
    "sr": 44100,                   # Sample rate
    "seed": 1100                   # Random seed
}

# -----------------------------
# PARTICIPANT METADATA
# -----------------------------
participant_info = {
    "participant_id": "P001",
    "session_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "headphones_used": False
}

# -----------------------------
# TONE GENERATION FUNCTIONS
# -----------------------------


def pure_tone(f, duration=0.5, sr=44100):
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return np.sin(2 * np.pi * f * t)


def play_tone(tone, sr=44100, max_volume=16000):
    audio = (tone * max_volume / np.max(np.abs(tone))).astype(np.int16)
    sa.play_buffer(audio, 1, 2, sr).wait_done()


# -----------------------------
# EXPERIMENT INITIALIZATION
# -----------------------------
rng = random.default_rng(seed=config["seed"])
f = config["base_freq"]
freq_diff = config["initial_diff"]
tone_time = config["tone_duration"]
stepf = config["step_diff"]
nReverse = config["n_reverse"]
max_volume = config["max_volume"]
sr = config["sr"]

nIncorrect = 0
accurate = 0
trial_counter = 0
trial_data_list = []

base_tone = pure_tone(f, tone_time, sr)

print("\n=== Auditory Frequency Discrimination Experiment ===")
print("**Use headphones for best results**\n")

# -----------------------------
# EXPERIMENT LOOP
# -----------------------------
while nIncorrect < nReverse:
    trial_counter += 1
    freq_position = rng.integers(1, 3)  # 1=low, 2=high

    test_tone = pure_tone(f - freq_diff if freq_position ==
                          1 else f + freq_diff, tone_time, sr)

    # Play tones
    play_tone(base_tone, sr, max_volume)
    time.sleep(config["silence_between_tones"])
    play_tone(test_tone, sr, max_volume)

    # Collect user response
    while True:
        try:
            resp = int(input("Which tone was higher? (1=first, 2=second): "))
            if resp in [1, 2]:
                break
            else:
                print("Enter 1 or 2")
        except ValueError:
            print("Enter a valid integer (1 or 2)")

    correct = int(resp == freq_position)
    if correct:
        accurate += 1
        if accurate == 2:
            freq_diff = max(freq_diff - stepf, 0.1)
            accurate = 0
    else:
        nIncorrect += 1
        freq_diff += stepf

    # Record trial data
    trial_data_list.append({
        "Trial": trial_counter,
        "Base_Frequency": f,
        "Frequency_Diff": freq_diff,
        "Freq_Position": freq_position,
        "Subject_Response": resp,
        "Correct": correct
    })

# -----------------------------
# DATA EXPORT & MERGE
# -----------------------------
df_session = pd.DataFrame(trial_data_list)
for key, val in participant_info.items():
    df_session[key] = val

csv_filename = f"auditory_test_{participant_info['participant_id']}.csv"

# Append to file if exists
if os.path.exists(csv_filename):
    df_previous = pd.read_csv(csv_filename)
    df_combined = pd.concat([df_previous, df_session], ignore_index=True)
else:
    df_combined = df_session.copy()

# Save full combined data
df_combined.to_csv(csv_filename, index=False)
print(f"\nData appended and saved to {csv_filename}")

# -----------------------------
# SUMMARY STATISTICS (CURRENT)
# -----------------------------
total_trials = len(df_session)
total_correct = df_session['Correct'].sum()
accuracy = round(total_correct / total_trials * 100, 2)
average_threshold = df_session['Frequency_Diff'].mean()
participant_info['session_date'] = datetime.now()
session_date = participant_info['session_date'].date()

# -----------------------------
# PLOT 1 — CURRENT SESSION ONLY
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
plt.title(
    f'Base Freq {f} Hz Session Performance (Accuracy: {accuracy}%)')
plt.suptitle(
    f'Participant {participant_info["participant_id"]}\nSession Date: {session_date}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    f'reports/{participant_info["participant_id"]}_performance_plot-{session_date}.png')
plt.show()

print("\n=== CURRENT SESSION SUMMARY ===")
print(f"Total Trials: {total_trials}")
print(f"Correct Responses: {total_correct}")
print(f"Accuracy: {accuracy}%")
print(f"Average Threshold: {average_threshold:.2f} Hz")


# -----------------------------
# PLOT 2 — HISTORICAL (ALL SESSIONS)
# -----------------------------
df_trials = pd.read_csv('auditory_test_P001.csv')

# Assign session numbers sequentially
df_trials['Session_Num'] = pd.factorize(df_trials['session_date'])[0] + 1

# Aggregate per session
session_summary = df_trials.groupby('Session_Num').agg(
    Accuracy=('Correct', 'mean'),               # Proportion correct
    Avg_Freq_Diff=('Frequency_Diff', 'mean')    # Average frequency difference
).reset_index()

# Convert accuracy to percentage
session_summary['Accuracy'] *= 100

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for accuracy
bars = ax1.bar(session_summary['Session_Num'], session_summary['Accuracy'],
               color='skyblue', edgecolor='black', label='Accuracy (%)')
ax1.set_xlabel('Session Number')
ax1.set_ylabel('Accuracy (%)', color='blue')
ax1.set_ylim(0, 100)
ax1.tick_params(axis='y', labelcolor='blue')

# Plot trend line for accuracy
z = np.polyfit(session_summary['Session_Num'], session_summary['Accuracy'], 1)
p = np.poly1d(z)
ax1.plot(session_summary['Session_Num'], p(session_summary['Session_Num']),
         color='blue', linestyle='--', label='Accuracy Trend')

# Secondary axis for average frequency difference
ax2 = ax1.twinx()
ax2.plot(session_summary['Session_Num'], session_summary['Avg_Freq_Diff'],
         color='red', marker='o', label='Avg Frequency Diff (Hz)')
ax2.set_ylabel('Average Frequency Difference (Hz)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title('Session Performance Overview')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(
    f'reports/{participant_info["participant_id"]}_overall_performance_plot.png')
plt.show()
