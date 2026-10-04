"""ローカルv2専用の試験環境へ渡す。未導入環境では速度のみ保留。"""
import io,json,math,os,subprocess,threading
from pathlib import Path
import soundfile as sf

ROOT=Path(__file__).parent
PYTHON=Path(os.environ.get('VOICE_ASR_PYTHON',str(ROOT/('.venv-asr-trial/Scripts/python.exe' if os.name=='nt' else '.venv-asr-trial/bin/python'))))
MODEL_ROOT=Path(os.environ.get('VOICE_ASR_MODEL_ROOT',str(ROOT/'docs/qa-2026-10-03/asr-models')))
MODEL=MODEL_ROOT/'models--Systran--faster-whisper-small/snapshots'
_slot=threading.BoundedSemaphore(1)


def measure(audio,sr,fixed_reading=False):
    unavailable=lambda reason:{'status':'unavailable','reason':reason}
    if not PYTHON.exists() or not any(MODEL.glob('*/model.bin')):
        return unavailable('asr_unavailable')
    if not _slot.acquire(blocking=False):
        return unavailable('asr_busy')
    try:
        output=io.BytesIO()
        sf.write(output,audio,sr,format='WAV',subtype='FLOAT')
        process=subprocess.run([str(PYTHON),str(ROOT/'local_speed_worker.py')]+(['--fixed-reading'] if fixed_reading else []),
            input=output.getvalue(),capture_output=True,timeout=45,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if process.returncode:
            return unavailable('recognition_failed')
        result=json.loads(process.stdout)
        if result.get('status')!='experimental':
            return unavailable(result.get('reason','recognition_failed'))
        rate=result.get('mora_per_second')
        if not isinstance(rate,(float,int)) or not math.isfinite(rate) or not 1<=rate<=20:
            return unavailable('recognition_failed')
        return {'status':'experimental','mora_per_second':float(rate)}
    except (ValueError,OSError,subprocess.TimeoutExpired):
        return unavailable('recognition_failed')
    finally:
        _slot.release()
