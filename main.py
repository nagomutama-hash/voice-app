from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import librosa
import io
import math
import json
import anthropic
from pydantic import BaseModel

app = FastAPI(title="声診断アプリ")

app.mount("/static", StaticFiles(directory="static"), name="static")

_ADVICE_SYSTEM = (
    "あなたはプロのボイストレーナーです。"
    "音声分析データをもとに、利用者へ励ましを込めた具体的なボイストレーニングアドバイスを日本語で提供してください。"
    "各セクションは2〜3文で簡潔にまとめてください。"
)


class AdviceRequest(BaseModel):
    min_hz: float
    max_hz: float
    mean_hz: float
    min_note: str
    max_note: str
    mean_note: str
    duration: float


@app.post("/advice")
async def generate_advice(req: AdviceRequest):
    try:
        semitones = (
            round(12 * math.log2(req.max_hz / req.min_hz), 1)
            if req.min_hz > 0 and req.max_hz > req.min_hz
            else 0
        )

        user_prompt = (
            f"以下の音声分析データをもとに、3つのセクションに分けてアドバイスをください。\n\n"
            f"【分析データ】\n"
            f"- 録音時間: {req.duration}秒\n"
            f"- 最低音: {req.min_note}（{req.min_hz} Hz）\n"
            f"- 最高音: {req.max_note}（{req.max_hz} Hz）\n"
            f"- 平均音程: {req.mean_note}（{req.mean_hz} Hz）\n"
            f"- 音域の幅: 約{semitones}半音\n\n"
            f"セクション:\n"
            f"1. range_characteristics（音域の特徴とその評価）\n"
            f"2. stability（声の安定性について）\n"
            f"3. improvement_tips（具体的な改善のヒントと練習法）"
        )

        client = anthropic.AsyncAnthropic()
        message = await client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": _ADVICE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "range_characteristics": {"type": "string"},
                            "stability": {"type": "string"},
                            "improvement_tips": {"type": "string"},
                        },
                        "required": ["range_characteristics", "stability", "improvement_tips"],
                        "additionalProperties": False,
                    },
                }
            },
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = next(b.text for b in message.content if b.type == "text")
        advice = json.loads(text)
        return JSONResponse({"success": True, "advice": advice})

    except Exception as e:
        import traceback

        return JSONResponse(
            {"success": False, "error": str(e), "detail": traceback.format_exc()},
            status_code=500,
        )


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        audio_data, sr = librosa.load(io.BytesIO(contents), sr=None, mono=True)

        duration = len(audio_data) / sr

        # 波形データ（表示用にダウンサンプリング）
        target_points = 1500
        step = max(1, len(audio_data) // target_points)
        waveform_samples = audio_data[::step]
        time_waveform = np.linspace(0, duration, len(waveform_samples))

        # 音程検出（pYIN アルゴリズム）
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )

        times_pitch = librosa.times_like(f0, sr=sr)
        pitch_hz = [float(v) if not np.isnan(v) else None for v in f0]

        # 統計情報
        voiced_f0 = [x for x in pitch_hz if x is not None]
        stats: dict = {"has_pitch": len(voiced_f0) > 0}
        if voiced_f0:
            min_hz = float(min(voiced_f0))
            max_hz = float(max(voiced_f0))
            mean_hz = float(np.mean(voiced_f0))
            stats.update(
                {
                    "min_hz": round(min_hz, 1),
                    "max_hz": round(max_hz, 1),
                    "mean_hz": round(mean_hz, 1),
                    "min_note": librosa.hz_to_note(min_hz),
                    "max_note": librosa.hz_to_note(max_hz),
                    "mean_note": librosa.hz_to_note(mean_hz),
                }
            )

        return JSONResponse(
            {
                "success": True,
                "waveform": {
                    "time": time_waveform.tolist(),
                    "amplitude": waveform_samples.tolist(),
                },
                "pitch": {
                    "time": times_pitch.tolist(),
                    "frequency": pitch_hz,
                },
                "duration": round(duration, 2),
                "sample_rate": int(sr),
                "stats": stats,
            }
        )

    except Exception as e:
        import traceback

        return JSONResponse(
            {"success": False, "error": str(e), "detail": traceback.format_exc()},
            status_code=500,
        )
