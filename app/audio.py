# -*- coding: utf-8 -*-

import numpy as np
import simpleaudio as sa

def pure_tone(f, duration, sr):
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return np.sin(2 * np.pi * f * t)

def play_tone(tone, sr, max_volume):
    audio = (tone * max_volume / np.max(np.abs(tone))).astype(np.int16)
    sa.play_buffer(audio, 1, 2, sr).wait_done()

def play_sequence(base, test, config):
    play_tone(base, config["sr"], config["max_volume"])
    import time
    time.sleep(config["silence_between_tones"])
    play_tone(test, config["sr"], config["max_volume"])