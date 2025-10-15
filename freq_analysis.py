import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime


df_trials = pd.read_csv('auditory_test_P001.csv')

trials = df_trials['Trial']
cumulative_correct = df_trials['Correct'].cumsum()
overall_accuracy = round(((max(cumulative_correct)/len(trials)) * 100), 2)


plt.figure(figsize=(10, 5))

plt.scatter(
    df_trials.loc[df_trials['Correct'] == 0, 'Trial'],
    df_trials.loc[df_trials['Correct'] == 0, 'Frequency_Diff'],
    color='red', edgecolors='black', s=100,
    label='Incorrect', zorder=5
)

# Plot correct responses
plt.scatter(
    df_trials.loc[df_trials['Correct'] == 1, 'Trial'],
    df_trials.loc[df_trials['Correct'] == 1, 'Frequency_Diff'],
    color='green', edgecolors='black', s=100,
    label='Correct', zorder=5
)

plt.plot(trials, df_trials['Frequency_Diff'],
         color='blue', linestyle='-', linewidth=1.5,
         label='Frequency Difference', zorder=3)
plt.xticks(range(1, df_trials['Trial'].max() + 1))
plt.xlabel('Trial')
plt.ylabel('Frequency Difference(Hz)')
plt.title(f'Performance Over Trials (Overall Accuracy: {overall_accuracy}%)')
plt.grid(True)
plt.legend(frameon=True)
plt.tight_layout()
plt.show()
