# -*- coding: utf-8 -*-

import numpy as np

class Experiment:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.f = self.config["base_freq"]
        self.freq_diff = self.config["initial_diff"]
        self.stepf = self.config["step_diff"]
        self.nReverse = self.config["n_reverse"]

        self.nIncorrect = 0
        self.accurate = 0
        self.trial_counter = 0
        self.data = []

    def next_trial(self):
        self.trial_counter += 1
        self.freq_position = np.random.randint(1, 3)

        test_freq = self.f - self.freq_diff if self.freq_position == 1 else self.f + self.freq_diff

        return {
            "trial": self.trial_counter,
            "freq_position": self.freq_position,
            "base_freq": self.f,
            "test_freq": test_freq,
            "freq_diff": self.freq_diff
        }

    def submit_answer(self, resp):
        correct = int(resp == self.freq_position)

        if correct:
            self.accurate += 1
            if self.accurate == 2:
                self.freq_diff = max(self.freq_diff - self.stepf, 0.1)
                self.accurate = 0
        else:
            self.nIncorrect += 1
            self.freq_diff += self.stepf

        self.data.append({
            "Trial": self.trial_counter,
            "Frequency_Diff": self.freq_diff,
            "Correct": correct
        })

        return correct

    def is_finished(self):
        return self.nIncorrect >= self.nReverse