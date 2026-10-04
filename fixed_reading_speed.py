"""Known reading length / detected speech span; no transcription model."""
from speech_timing import speech_timing


def measure(audio, sr, prompt_id, completed, reading):
    if prompt_id != reading['id'] or not completed:
        return {'status': 'unavailable', 'reason': 'reading_not_confirmed'}
    timing, _, _ = speech_timing(audio, sr)
    span = timing['span_seconds']
    if span < 3:
        return {'status': 'unavailable', 'reason': 'insufficient_speech'}
    rate = reading['mora_count'] / span
    if not 1 <= rate <= 20:
        return {'status': 'unavailable', 'reason': 'reading_uncertain'}
    return {'status': 'experimental', 'mora_per_second': rate}
