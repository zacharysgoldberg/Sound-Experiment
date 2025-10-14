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
    "base_freq": 600,             # Base frequency in Hz
    "initial_diff": 5,            # Initial frequency difference
    "tone_duration": 0.5,         # Duration of each tone in seconds
    "step_diff": 0.2,             # Step size for staircase adjustment
    "n_reverse": 3,               # Number of allowed incorrect trials
    "silence_between_tones": 1.0,  # Seconds of silence between tones
    "max_volume": 16000,          # Max amplitude for tones
    "sr": 44100,                  # Sample rate
    "seed": 1100                  # Random seed
}

# -----------------------------
# PARTICIPANT METADATA
# -----------------------------
participant_info = {
    "participant_id": "P001",
    "session_date": datetime.now(),
    "headphones_used": False
}

# -----------------------------
# TONE GENERATION FUNCTIONS
# -----------------------------


def pure_tone(f, duration=0.5, sr=44100):
    t = np.linspace(0, duration, int(duration*sr), endpoint=False)
    tone = np.sin(2 * np.pi * f * t)
    return tone


def play_tone(tone, sr=44100, max_volume=16000):
    # Scale tone once and ensure contiguous int16 array
    audio = (tone * max_volume / np.max(np.abs(tone))).astype(np.int16)
    audio = np.ascontiguousarray(audio)
    sa.play_buffer(audio, 1, 2, sr).wait_done()


# -----------------------------
# EXPERIMENT INITIALIZATION
# -----------------------------
rng = random.default_rng(seed=config["seed"])
f = config["base_freq"]
df = config["initial_diff"]
tone_time = config["tone_duration"]
stepf = config["step_diff"]
nReverse = config["n_reverse"]
max_volume = config["max_volume"]
sr = config["sr"]

nIncorrect = 0
accurate = 0
trial_counter = 0
trial_data_list = []

# Generate base tone
base_tone = pure_tone(f, tone_time, sr)

print("\n=== Auditory Frequency Discrimination Experiment ===")
print("**Use headphones for best results**\n")

# -----------------------------
# EXPERIMENT LOOP
# -----------------------------
while nIncorrect < nReverse:
    trial_counter += 1
    freq_position = rng.integers(1, 3)  # 1=low, 2=high

    # Generate test tone
    if freq_position == 1:
        test_tone = pure_tone(f - df, tone_time, sr)
    else:
        test_tone = pure_tone(f + df, tone_time, sr)

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

    # Check correctness
    correct = int(resp == freq_position)
    if correct:
        accurate += 1
        if accurate == 2:
            df = max(df - stepf, 0.1)
            accurate = 0
    else:
        nIncorrect += 1
        df += stepf

    # Record trial data
    trial_data_list.append({
        "Trial": trial_counter,
        "Base_Frequency": f,
        "Frequency_Diff": df,
        "Freq_Position": freq_position,
        "Subject_Response": resp,
        "Correct": correct
    })

# -----------------------------
# DATA ANALYSIS & EXPORT
# -----------------------------
df_trials = pd.DataFrame(trial_data_list)

# Merge participant metadata
for key, val in participant_info.items():
    df_trials[key] = val

csv_filename = f"auditory_test_{participant_info['participant_id']}.csv"
df_trials.to_csv(csv_filename, index=False)
print(f"\nExperiment complete. Data saved to CSV: {csv_filename}")

# -----------------------------
# PLOTTING
# -----------------------------
plt.figure(figsize=(10, 5))

# Plot the staircase line
plt.plot(df_trials['Trial'], df_trials['Frequency_Diff'],
         marker='o', label='Frequency Difference', color='blue')

# Plot correct/incorrect markers slightly above the line
y_offset = df_trials['Frequency_Diff'] * 0.02  # 2% above line
plt.scatter(df_trials['Trial'], df_trials['Frequency_Diff'] + y_offset,
            c=df_trials['Correct'].map({0: 'red', 1: 'green'}),
            label='Correct (green) / Incorrect (red)', s=100, zorder=5, edgecolors='black')

plt.xlabel('Trial')
plt.ylabel('Frequency Difference (Hz)')
plt.title(f'Frequency Discrimination Staircase for {f} Hz Base Tone')
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# SUMMARY STATISTICS
# -----------------------------
total_trials = len(df_trials)
total_correct = df_trials['Correct'].sum()
overall_accuracy = total_correct / total_trials * 100
average_threshold = df_trials['Frequency_Diff'].mean()

print("\n=== SUMMARY STATISTICS ===")
print(f"Total Trials: {total_trials}")
print(f"Total Correct Responses: {total_correct}")
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
print(f"Average Frequency Difference Threshold: {average_threshold:.2f} Hz")

# Cumulative performance plot
df_trials['Cumulative_Correct'] = df_trials['Correct'].cumsum()

plt.figure(figsize=(10, 5))
plt.plot(df_trials['Trial'], df_trials['Cumulative_Correct'],
         marker='o', color='blue', label='Cumulative Correct')
plt.xlabel('Trial')
plt.ylabel('Cumulative Correct Responses')
plt.title('Cumulative Performance Over Trials')
plt.grid(True)
plt.legend()
plt.show()
