import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import simpleaudio as sa
import time

# A pure tone


def pure_tone(f, time=0.5, sr=44100):
    time_vec = np.linspace(0, time, int(time * sr))
    test_note = np.sin(f * time_vec * 2 * np.pi)
    return test_note


f = 300
sr = 44100
tone_time = 0.5
tone = pure_tone(f, tone_time)
tone = tone * 16000 / np.max(np.abs(tone))
tone = tone.astype(int)
sa.play_buffer(tone, 1, 2, sr)
print(tone)


# Two tones seperated by silece

def silent_tone(f, df, time, sr=44100):
    time_vec = np.linspace(0, time, int(time * sr))
    scale = np.array([])
    silent_interval = 0.75
    nsamples_silence = int(silent_interval * sr)
    silence = np.zeros(nsamples_silence)
    high_f = f + df / 2
    low_f = f - df / 2
    high_tone = np.sin(high_f * time_vec * 2 * np.pi)
    low_tone = np.sin(low_f * time_vec * 2 * np.pi)
    h = np.concatenate((high_tone, silence))
    l = np.concatenate((low_tone, silence))
    scale = np.concatenate((scale, l, h))
    return scale


f = 300
tone_time = 0.5
df = 2
max_amplitude = 8000
sr = 44100
scale = silent_tone(f, df, tone_time)
scale = scale * max_amplitude / np.max(np.abs(scale))
scale = scale.astype(np.int16)
sa.play_buffer(scale, 1, 2, sr)

print('Silent Tone: ', scale)


# Generating random tone

rng = random.default_rng(seed=1100)
f = 600
df = 5
tone_time = 0.5
stepf = 0.2
nReverse = 3
max_volume = 16000
sr = 44100
correct = []
subject_response = []
trial_accuracy = []
staircase = []
nIncorrect = 0     # Incorrect response tracker
accurate = 0

base_tone = pure_tone(f, tone_time, sr=sr)
# reduce the volume for safety.
base_tone = base_tone * max_volume / np.max(np.abs(base_tone))
base_tone = base_tone.astype(int)
print('Pure Tone: ', base_tone)

while nIncorrect < nReverse:
    freq_position = rng.integers(1, 3)
    if freq_position == 1:
        test_tone = pure_tone(f - df, tone_time, sr=sr)
        test_tone = test_tone * max_volume / np.max(np.abs(test_tone))
        test_tone = test_tone.astype(int)
    else:
        test_tone = pure_tone(f + df, tone_time, sr=sr)
        test_tone = test_tone * max_volume / np.max(np.abs(test_tone))
        test_tone = test_tone.astype(int)
    play_tone = sa.play_buffer(base_tone, 1, 2, sr)
    play_tone.wait_done()

    time.sleep(2)

    play_tone = sa.play_buffer(test_tone, 1, 2, sr)
    play_tone.wait_done()
    print('Which tone was the higher frequency?')
    print('Enter 1 for the first tone or 2 for the second tone')
    while True:
        trial_response = int(input())
        if trial_response == 2 or trial_response == 1:
            break
        else:
            print('Enter a valid input')
    if trial_response == freq_position:
        accurate += 1
    else:
        nIncorrect += 1
        df += stepf
    if accurate == 2:
        trial_accuracy = 0
        df -= stepf
    staircase.append(df)
    trial_accuracy.append(accurate)
    correct.append(freq_position)
    subject_response.append(trial_response)

print('Answer Key:', correct)
print('Subject responses:', subject_response)
print('Total accurate:', accurate)

# %%%
plt.plot(staircase)
plt.xlabel('Trials')
plt.ylabel('Difference in Frequency(Hz)')
plt.title('Frequency is ' + str(f) + ' Hz')
plt.show()
# %%
