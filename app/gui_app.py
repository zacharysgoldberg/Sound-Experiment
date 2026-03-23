import tkinter as tk
from tkinter import messagebox, ttk
import threading
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from experiment import Experiment
from audio import pure_tone, play_sequence
from data_manager import save_session
from config_paths import DATA_DIR
from analysis.analysis import plot_data

# CONFIG
config = {
    "base_freq": 600,
    "initial_diff": 5,
    "tone_duration": 0.5,
    "step_diff": 0.2,
    "n_reverse": 3,
    "silence_between_tones": 1.0,
    "max_volume": 16000,
    "sr": 44100
}


class App:
    def __init__(self, root):
        self.root = root
        self.exp = Experiment(config)

        # Label
        self.label = tk.Label(root, text="Select or Enter Participant Name")
        self.label.pack(pady=5)

        # Frame for dropdown + entry
        frame = tk.Frame(root)
        frame.pack()

        # Get existing participant IDs
        existing_ids = self.get_existing_participants()

        # Combobox with existing participants
        self.participant_combo = ttk.Combobox(frame, values=existing_ids)
        self.participant_combo.pack(side="left", padx=5)
        self.participant_combo.set("Select...")

        # Entry for new participant
        self.entry = tk.Entry(frame, fg="gray")
        self.entry.insert(0, "Enter new...")
        self.entry.pack(side="left", padx=5)
        
        # Add focus in/out behavior to mimic placeholder
        self.entry.bind("<FocusIn>", self.on_entry_click)
        self.entry.bind("<FocusOut>", self.on_focusout)

        # Start button
        # Create the button initially disabled
        self.start_btn = tk.Button(root, text="Start", command=self.start, state="disabled")
        self.start_btn.pack(pady=5)
        
        # Enable start button when either combobox or entry changes
        self.participant_combo.bind("<<ComboboxSelected>>", self.check_start_ready)
        self.entry.bind("<KeyRelease>", self.check_start_ready)

        # Frame for first/second buttons horizontally
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.btn1 = tk.Button(btn_frame, text="1", command=lambda: self.answer(1), state="disabled", width=10)
        self.btn2 = tk.Button(btn_frame, text="2", command=lambda: self.answer(2), state="disabled", width=10)
        self.btn1.pack(side="left", padx=10)
        self.btn2.pack(side="left", padx=10)
        
    def on_entry_click(self, event):
        if self.entry.get() == "Enter new...":
            self.entry.delete(0, "end")
            self.entry.config(fg="black")

    def on_focusout(self, event):
        if self.entry.get() == "":
            self.entry.insert(0, "Enter new...")
            self.entry.config(fg="gray")
    
    def check_start_ready(self, event=None):
        typed = self.entry.get().strip()
        selected = self.participant_combo.get().strip()
        if typed and typed != "Enter new":
            self.start_btn.config(state="normal", fg="black")
        elif selected and selected != "Select...":
            self.start_btn.config(state="normal", fg="black")
        else:
            self.start_btn.config(state="disabled", fg="gray")

    def get_existing_participants(self):
        if not os.path.exists(DATA_DIR):
            return []
        files = os.listdir(DATA_DIR)
        ids = []
        for f in files:
            if f.startswith("auditory_test_") and f.endswith(".csv"):
                pid = f.replace("auditory_test_", "").replace(".csv", "")
                ids.append(pid)
        return ids

    def start(self):
        # Use entry if typed, else combobox
        typed_id = self.entry.get().strip()
        selected_id = self.participant_combo.get().strip()
        if typed_id:
            self.participant_id = typed_id
        elif selected_id and selected_id != "Select...":
            self.participant_id = selected_id
        else:
            messagebox.showerror("Error", "Enter or select a Participant Name")
            return

        self.start_btn.config(state="disabled")
        self.entry.config(state="disabled")
        self.participant_combo.config(state="disabled")

        self.next_trial()

    def next_trial(self):
        if self.exp.is_finished():
            self.finish()
            return

        trial = self.exp.next_trial()

        base = pure_tone(trial["base_freq"], config["tone_duration"], config["sr"])
        test = pure_tone(trial["test_freq"], config["tone_duration"], config["sr"])

        self.label.config(text=f"Trial {trial['trial']}")

        threading.Thread(target=self.play_audio, args=(base, test)).start()

    def play_audio(self, base, test):
        play_sequence(base, test, config)

        self.label.config(text="Which tone was higher?")
        self.btn1.config(state="normal")
        self.btn2.config(state="normal")

    def answer(self, resp):
        self.btn1.config(state="disabled")
        self.btn2.config(state="disabled")

        self.exp.submit_answer(resp)

        self.root.after(300, self.next_trial)

    def finish(self):
        # Save session
        path = save_session(self.exp.data, self.participant_id)
        messagebox.showinfo("Done", f"Saved to {path}")

        # Run analysis for this participant
        plot_data(self.participant_id)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Auditory Experiment")
    width = 300
    height = 150

    # Get screen size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Compute center position
    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))

    # Set geometry
    root.geometry(f"{width}x{height}+{x}+{y}")
    app = App(root)
    root.mainloop()