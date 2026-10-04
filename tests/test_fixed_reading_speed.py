import numpy as np
import pytest
from fixed_reading_speed import measure

READING = {'id': 'fixed', 'mora_count': 41}


def tone(seconds):
    t = np.arange(round(seconds * 16000)) / 16000
    return .1 * np.sin(2 * np.pi * 150 * t)


def test_padding_is_excluded_and_internal_pause_is_included():
    audio = np.concatenate([np.zeros(16000), tone(3), np.zeros(8000), tone(3), np.zeros(16000)])
    result = measure(audio, 16000, 'fixed', True, READING)
    assert result['mora_per_second'] == pytest.approx(41 / 6.5)


def test_wrong_prompt_unconfirmed_silence_and_short_speech_are_withheld():
    for audio, prompt, completed in [(tone(6), 'other', True), (tone(6), 'fixed', False),
                                     (np.zeros(96000), 'fixed', True), (tone(2), 'fixed', True)]:
        assert measure(audio, 16000, prompt, completed, READING)['status'] == 'unavailable'
