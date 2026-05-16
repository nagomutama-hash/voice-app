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
        "あなたはボイストレーニング専門家・玉井の声診断AIアシスタントです。\n"
        "音声分析データをもとに、利用者がZoom無料声診断を受けたくなるコメントを日本語で生成してください。\n\n"
        "【ゴール】\n"
        "読んだ人が『この声のことをもっと知りたい！玉井さんに診てもらいたい！』と感じること。\n\n"
        "【各セクションの役割】\n"
        "1. voice_character：声の個性・魅力を具体的に描写して『自分の声ってそんな特徴があるの！』と気づかせる\n"
        "2. potential：可能性の入口だけ見せて焦らす。改善のキーワード（例：共鳴・芯・表情）は出すが、方法は教えない。『答えはあなたの声を実際に聴かないとわかりません』で締める\n"
        "3. next_step：Zoom無料声診断で何が得られるかを魅力的に伝えて申込みへ誘導する\n\n"
        "【ルール】\n"
        "・具体的な練習法・エクササイズ・トレーニング手順は書かない\n"
        "・改善の『キーワード』は出してよいが『やり方』は教えない\n"
        "・各セクション3〜4文。温かく、背中を押すトーンで。\n\n"
        "=== ボイストレーニング専門知識 ===\n\n"
        + knowledge
        + "\n\n必ず以下のJSON形式のみで回答してください（他のテキストは不要）:\n"
        '{"voice_character": "...", "potential": "...", "next_step": "..."}'
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
            f"以下の音声分析データをもとに、3つのセクションで声診断コメントを生成してください。\n\n"
            f"【分析データ】\n"
            f"- 録音時間: {req.duration}秒\n"
            f"- 最低音: {req.min_note}（{req.min_hz} Hz）\n"
            f"- 最高音: {req.max_note}（{req.max_hz} Hz）\n"
            f"- 平均音程: {req.mean_note}（{req.mean_hz} Hz）\n"
            f"- 音域の幅: 約{semitones}半音\n\n"
            f"セクション:\n"
            f"1. voice_character：この方の声の個性・魅力を具体的に描写する。音域・音程・声質を言葉で表現し『自分の声ってそんな特徴があるの！』と気づかせる。褒めて、声への関心を高める。\n"
            f"2. potential：この声が持つ可能性と課題の『入口』だけを見せる。『共鳴』『芯』『表情』『息の支え』などのキーワードは使ってよいが、具体的なやり方・練習法は絶対に書かない。「ただし、その方法はあなたの声の構造によって異なるため、実際の声を聴かないと判断できません」という流れで締める。\n"
            f"3. next_step：玉井の20分Zoom無料声診断で『あなただけの声の設計図』が見えてくることを伝える。診断を受けることで何がわかるか・どう変わるかを魅力的に描写し、申込みへの一歩を後押しする言葉で締める。"
        )

        client = anthropic.AsyncAnthropic(api_key=_API_KEY)
        message = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": _build_advice_system(),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = next(b.text for b in message.content if b.type == "text")
        # JSON部分だけを抽出（マークダウンコードブロックにも対応）
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            raise ValueError(f"JSONが見つかりません: {text[:200]}")
        advice = json.loads(json_match.group())
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
