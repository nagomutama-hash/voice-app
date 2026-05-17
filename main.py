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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
        "1. voice_character：音域・音程の揺れ・声の強弱・明るさ・発声密度のデータをすべて組み合わせて、その人固有の声の個性・魅力を描写する。データが少しでも違えば必ず違う描写になるよう、具体的な数値的特徴を言葉に変換して『自分の声ってそんな特徴があるの！』と気づかせる\n"
        "2. potential：その声のデータが示す具体的な課題の入口を見せて焦らす。共鳴・芯・表情・息の支えなどのキーワードは出すが方法は教えない。『答えはあなたの声を実際に聴かないとわかりません』で締める\n"
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
    # 追加の声質データ（デフォルト0で後方互換性を維持）
    pitch_std: float = 0.0      # F0標準偏差 Hz（音程の揺れ・抑揚の幅）
    voiced_ratio: float = 0.0   # 有声区間の割合 %
    rms_cv: float = 0.0         # 音量変動係数 %（強弱の幅）
    brightness_hz: float = 0.0  # スペクトル重心 Hz（声の明るさ）
    harmonic_ratio: float = 0.0  # 倍音比率 %（声の響き・共鳴）
    speech_rate: float = 0.0     # 発話速度 フレーズ/秒（話す速さ）
    rms_trend: float = 0.0       # 音量傾向 %（正=後半強・負=後半フェード）


@app.post("/advice")
async def generate_advice(req: AdviceRequest):
    semitones = (
        round(12 * math.log2(req.max_hz / req.min_hz), 1)
        if req.min_hz > 0 and req.max_hz > req.min_hz
        else 0
    )

    # 声質データの解釈ラベル（Claudeが多彩な診断を出すための手がかり）
    pitch_cv = (req.pitch_std / req.mean_hz * 100) if req.mean_hz > 0 else 0.0
    if pitch_cv < 5:
        pitch_label = "抑揚が控えめ・平坦な傾向（単調になりやすい）"
    elif pitch_cv < 15:
        pitch_label = "自然な抑揚の波がある"
    elif pitch_cv < 25:
        pitch_label = "音程の変化が豊か・声の表情が大きい"
    else:
        pitch_label = "音程の揺れが目立つ（ビブラートまたは不安定）"

    if req.rms_cv < 20:
        dynamic_label = "音量がほぼ一定・強弱が少ない（声のメリハリ不足の可能性）"
    elif req.rms_cv < 45:
        dynamic_label = "適度な強弱のコントラストがある"
    else:
        dynamic_label = "強弱の変化が大きい・メリハリのある発声"

    if req.voiced_ratio < 35:
        voiced_label = f"息継ぎや間が多め（発声率{req.voiced_ratio:.0f}%）"
    elif req.voiced_ratio < 65:
        voiced_label = f"声と間のバランスが良い（発声率{req.voiced_ratio:.0f}%）"
    else:
        voiced_label = f"連続した発声が多い・間が少ない（発声率{req.voiced_ratio:.0f}%）"

    if req.brightness_hz < 700:
        brightness_label = "低音成分が豊か・深みと重さのある声質"
    elif req.brightness_hz < 1300:
        brightness_label = "バランスの取れた声の明るさ"
    else:
        brightness_label = "高音成分が強い・明るく軽やかな声質"

    if req.harmonic_ratio < 40:
        harmonic_label = "息混じりの声・倍音が少なめ（声の通りにくさの可能性）"
    elif req.harmonic_ratio < 65:
        harmonic_label = "響きのバランスが取れている"
    else:
        harmonic_label = "倍音が豊か・よく響く声質"

    if req.speech_rate < 0.8:
        rate_label = "ゆっくりめの話し方・間が多い（丁寧な印象・もたつき感の可能性）"
    elif req.speech_rate < 2.0:
        rate_label = "自然なテンポで話している"
    else:
        rate_label = "テンポが速め・フレーズが細かい（早口気味の可能性）"

    if req.rms_trend < -20:
        trend_label = "後半に向かって声が弱まる傾向（息の支えが課題の可能性）"
    elif req.rms_trend < 20:
        trend_label = "音量が安定して持続している"
    else:
        trend_label = "後半に向かって声が強まる・気持ちが乗ってくる傾向"

    user_prompt = (
        f"以下の音声分析データをもとに、3つのセクションで声診断コメントを生成してください。\n\n"
        f"【基本データ】\n"
        f"- 録音時間: {req.duration}秒\n"
        f"- 最低音: {req.min_note}（{req.min_hz} Hz）\n"
        f"- 最高音: {req.max_note}（{req.max_hz} Hz）\n"
        f"- 平均音程: {req.mean_note}（{req.mean_hz} Hz）\n"
        f"- 音域の幅: 約{semitones}半音\n\n"
        f"【声質の特徴】\n"
        f"- 抑揚・音程の揺れ: {pitch_label}\n"
        f"- 声の強弱: {dynamic_label}\n"
        f"- 発声の密度: {voiced_label}\n"
        f"- 声の明るさ・声質: {brightness_label}\n"
        f"- 声の響き・倍音: {harmonic_label}\n"
        f"- 話す速さ・テンポ: {rate_label}\n"
        f"- 声の持続力・傾向: {trend_label}\n\n"
        f"セクション:\n"
        f"1. voice_character：この方の声の個性・魅力を具体的に描写する。基本データと声質の特徴を両方使って、この声ならではの個性を言葉にする。褒めて声への関心を高める。\n"
        f"2. potential：この声のデータが示す課題の『入口』だけを見せる。『共鳴』『芯』『表情』『息の支え』などのキーワードは使ってよいが、具体的なやり方・練習法は絶対に書かない。「ただし、その方法はあなたの声の構造によって異なるため、実際の声を聴かないと判断できません」という流れで締める。\n"
        f"3. next_step：玉井の20分Zoom無料声診断で『あなただけの声の設計図』が見えてくることを伝える。診断を受けることで何がわかるか・どう変わるかを魅力的に描写し、申込みへの一歩を後押しする言葉で締める。"
    )

    async def event_stream():
        import re, traceback
        try:
            client = anthropic.AsyncAnthropic(api_key=_API_KEY)
            full_text = ""
            async with client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=[{
                    "type": "text",
                    "text": _build_advice_system(),
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    full_text += text
                    yield f"data: {json.dumps({'type': 'chunk', 'text': text})}\n\n"

            json_match = re.search(r'\{.*\}', full_text, re.DOTALL)
            if not json_match:
                raise ValueError(f"JSONが見つかりません: {full_text[:200]}")
            advice = json.loads(json_match.group())
            yield f"data: {json.dumps({'type': 'done', 'advice': advice})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'detail': traceback.format_exc()})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def root():
    return FileResponse(str(Path(__file__).resolve().parent / "static" / "index.html"))


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # 16 kHz にダウンサンプリング：pyin の計算量を大幅削減
        audio_data, sr = librosa.load(io.BytesIO(contents), sr=16000, mono=True)
        duration = len(audio_data) / sr

        target_points = 1500
        step = max(1, len(audio_data) // target_points)
        waveform_samples = audio_data[::step]
        time_waveform = np.linspace(0, duration, len(waveform_samples))

        # yin は pyin より約500倍高速（確率的 HMM なし）
        # 無音フレームは fmax 超の値を返すので範囲フィルタで除去
        _fmin = librosa.note_to_hz("C2")
        _fmax = librosa.note_to_hz("C7")
        f0 = librosa.yin(
            audio_data,
            fmin=_fmin,
            fmax=_fmax,
            sr=sr,
            frame_length=1024,
            hop_length=256,
        )
        times_pitch = librosa.times_like(f0, sr=sr, hop_length=256)
        pitch_hz = [float(v) if _fmin < v <= _fmax else None for v in f0]

        voiced_f0 = [x for x in pitch_hz if x is not None]
        stats: dict = {"has_pitch": len(voiced_f0) > 0}
        if voiced_f0:
            min_hz = float(min(voiced_f0))
            max_hz = float(max(voiced_f0))
            mean_hz = float(np.mean(voiced_f0))

            # 追加分析① 音程の揺れ（抑揚の幅）
            pitch_std = round(float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0, 1)

            # 追加分析② 有声率（声を出している割合）
            voiced_ratio = round(len(voiced_f0) / len(pitch_hz) * 100, 1) if len(pitch_hz) > 0 else 0.0

            # 追加分析③ 音量の強弱（変動係数）
            rms = librosa.feature.rms(y=audio_data, hop_length=256)[0]
            rms_mean = float(np.mean(rms))
            rms_cv = round(float(np.std(rms) / rms_mean * 100), 1) if rms_mean > 0 else 0.0

            # 追加分析④ 声の明るさ（スペクトル重心）
            spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=256)[0]
            brightness_hz = round(float(np.mean(spec_centroid)), 1)

            # 追加分析⑤ 声の響き（倍音比率）
            harmonic = librosa.effects.harmonic(audio_data)
            harmonic_energy = float(np.mean(harmonic ** 2))
            total_energy = float(np.mean(audio_data ** 2))
            harmonic_ratio = round(harmonic_energy / total_energy * 100, 1) if total_energy > 0 else 0.0

            # 追加分析⑥ 話す速さ（発声セグメント開始回数 / 秒）
            voiced_flags = np.array([1 if x is not None else 0 for x in pitch_hz])
            transitions = int(np.sum(np.diff(voiced_flags) > 0))
            speech_rate = round(transitions / duration, 2) if duration > 0 else 0.0

            # 追加分析⑦ 声の持続力（RMS傾向）
            if len(rms) > 1:
                x_trend = np.linspace(0, 1, len(rms))
                slope = float(np.polyfit(x_trend, rms, 1)[0])
                rms_trend = round(slope / rms_mean * 100, 1) if rms_mean > 0 else 0.0
            else:
                rms_trend = 0.0

            stats.update({
                "min_hz": round(min_hz, 1),
                "max_hz": round(max_hz, 1),
                "mean_hz": round(mean_hz, 1),
                "min_note": librosa.hz_to_note(min_hz),
                "max_note": librosa.hz_to_note(max_hz),
                "mean_note": librosa.hz_to_note(mean_hz),
                "pitch_std": pitch_std,
                "voiced_ratio": voiced_ratio,
                "rms_cv": rms_cv,
                "brightness_hz": brightness_hz,
                "harmonic_ratio": harmonic_ratio,
                "speech_rate": speech_rate,
                "rms_trend": rms_trend,
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
