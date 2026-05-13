import io
import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
import librosa
import numpy as np
import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# .envファイルからAPIキーを読み込む（ローカル開発用）
_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)

# 環境変数またはファイルからAPIキーを取得
_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not _API_KEY and _ENV_FILE.exists():
    for _line in _ENV_FILE.read_bytes().decode("utf-8-sig").splitlines():
        if "ANTHROPIC_API_KEY=" in _line:
            _API_KEY = _line.split("=", 1)[1].strip()
            break

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
_knowledge_text: str = ""


def _load_pdf_knowledge() -> str:
    sections: list[str] = []
    for pdf_path in sorted(KNOWLEDGE_DIR.glob("*.pdf")):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = [p for page in pdf.pages if (p := page.extract_text())]
                if pages:
                    sections.append(f"【{pdf_path.stem}】\n" + "\n".join(pages))
        except Exception as exc:
            print(f"PDF読み込みエラー [{pdf_path.name}]: {exc}")
    return "\n\n---\n\n".join(sections)


def _build_advice_system() -> str:
    knowledge = _knowledge_text or "（専門知識ファイルが見つかりませんでした。一般的な知識でアドバイスしてください。）"
    return (
        "あなたはボイストレーニング専門家です。"
        "以下の専門教材（PDF）の知識を活かして、音声分析データをもとに利用者へ励ましを込めた具体的なアドバイスを日本語で提供してください。"
        "各セクションは2〜3文で簡潔にまとめてください。\n\n"
        "=== ボイストレーニング専門知識 ===\n\n"
        + knowledge
        + "\n\n必ず以下のJSON形式のみで回答してください（他のテキストは不要）:\n"
        '{"range_characteristics": "...", "stability": "...", "improvement_tips": "..."}'
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _knowledge_text
    _knowledge_text = _load_pdf_knowledge()
    pdf_count = len(list(KNOWLEDGE_DIR.glob("*.pdf")))
    print(f"PDF知識読み込み完了: {pdf_count}ファイル / {len(_knowledge_text)}文字")
    print(f"APIキー: {'設定済み' if _API_KEY else '未設定！'}")
    yield


app = FastAPI(title="声診断アプリ", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")


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

        client = anthropic.AsyncAnthropic(api_key=_API_KEY)
        message = await client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": _build_advice_system(),
                "cache_control": {"type": "ephemeral"},
            }],
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
    return FileResponse(str(Path(__file__).resolve().parent / "static" / "index.html"))


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        audio_data, sr = librosa.load(io.BytesIO(contents), sr=None, mono=True)
        duration = len(audio_data) / sr

        target_points = 1500
        step = max(1, len(audio_data) // target_points)
        waveform_samples = audio_data[::step]
        time_waveform = np.linspace(0, duration, len(waveform_samples))

        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        times_pitch = librosa.times_like(f0, sr=sr)
        pitch_hz = [float(v) if not np.isnan(v) else None for v in f0]

        voiced_f0 = [x for x in pitch_hz if x is not None]
        stats: dict = {"has_pitch": len(voiced_f0) > 0}
        if voiced_f0:
            min_hz = float(min(voiced_f0))
            max_hz = float(max(voiced_f0))
            mean_hz = float(np.mean(voiced_f0))
            stats.update({
                "min_hz": round(min_hz, 1),
                "max_hz": round(max_hz, 1),
                "mean_hz": round(mean_hz, 1),
                "min_note": librosa.hz_to_note(min_hz),
                "max_note": librosa.hz_to_note(max_hz),
                "mean_note": librosa.hz_to_note(mean_hz),
            })

        return JSONResponse({
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
        })

    except Exception as e:
        import traceback
        return JSONResponse(
            {"success": False, "error": str(e), "detail": traceback.format_exc()},
            status_code=500,
        )
