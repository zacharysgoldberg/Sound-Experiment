# Sound-Experiment

An auditory test designed to measure the minimum frequency difference required for reliable discrimination. This experiment demonstrates principles of **Weber’s Law** in auditory perception.

---

## Background

Weber’s Law states that _the just-noticeable difference in a stimulus is a constant proportion of the original stimulus_. This experiment explores how frequency differences are perceived by the human auditory system.

---

## Requirements

- Python 3.8 or higher
- `virtualenv` or Python venv module

> **Windows users:** May need Microsoft Visual C++ Build Tools installed before installing dependencies.

### Setup

1. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```
3. **Running the Experiment**
   ```
   python freq_test.py
   ```

## Output

- User responses are collected via terminal input during the experiment.
- At the end of the session, a **graphical visualization** is displayed showing the progression of frequency differences across trials:
  - **Blue line:** Frequency difference (staircase) for each trial.
  - **Green markers:** Correct responses.
  - **Red markers:** Incorrect responses.
- A **cumulative performance plot** is also shown to illustrate learning or trends over the trials.
- The program calculates **summary statistics**, including:
  - Overall accuracy (% correct responses)
  - Average frequency difference threshold
- All trial data and participant metadata are automatically saved to a **CSV file** for further analysis.

## Notes

- Ensure volume is set to a comfortable level before starting.
- Headphones are highly recommended to prevent external noise interference and to clearly perceive the tones.
