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
   python freq_sounds.py
   ```

## Output

- The program collects user responses via terminal input.
- At the end of the experiment, a graph is displayed showing the progression of frequency differences over trials.
- Frequency adjustments follow a staircase method to measure perceptual thresholds.

## Notes

- Ensure volume is set to a comfortable level before starting.
- Headphones are highly recommended to prevent external noise interference and to clearly perceive the tones.
