import io
import json
import math
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
import librosa
import numpy as np
import soundfile as sf
import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from diagnosis import CONFIG as DIAGNOSIS_CONFIG, build_diagnosis
from asr_speed import measure as measure_speech_speed
from fixed_reading_speed import measure as measure_fixed_reading_speed
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from test_entry import TEST_PREFIX, entry_access_allowed, render_entry_html
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
    for txt_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        try:
            text = txt_path.read_text(encoding="utf-8")
            if text.strip():
                sections.append(f"【{txt_path.stem}】\n" + text)
        except Exception as exc:
            print(f"テキスト読み込みエラー [{txt_path.name}]: {exc}")
    return "\n\n---\n\n".join(sections)


def _build_advice_system() -> str:
    knowledge = _knowledge_text or "（専門知識ファイルが見つかりませんでした。一般的な知識でアドバイスしてください。）"
    return (
        "あなたはボイストレーニング専門家の声診断アシスタントです。\n"
        "音声分析データをもとに、利用者がZoom無料声診断を受けたくなるコメントを日本語で生成してください。\n\n"
        "【ゴール】\n"
        "読んだ人が『この声のことをもっと知りたい！プロに診てもらいたい！』と感じること。\n\n"
        "【各セクションの役割】\n"
        "1. voice_character：音域・音程の揺れ・声の強弱・明るさ・発声密度のデータをすべて組み合わせて、その人固有の声の個性・魅力を描写する。データが少しでも違えば必ず違う描写になるよう、具体的な数値的特徴を言葉に変換して『自分の声ってそんな特徴があるの！』と気づかせる\n"
        "2. potential：その声のデータが示す具体的な課題の入口を見せて焦らす。共鳴・芯・表情・息の支えなどのキーワードは出すが方法は教えない。「実際の声をお聴きすれば、さらに正確な診断とあなただけのアドバイスをお伝えできます」という前向きな一文で締める\n"
        "3. next_step：Zoom無料声診断で何が得られるかを魅力的に伝えて申込みへ誘導する。人名・固有名詞は一切使わず『プロの声診断』『専門家』などの表現にとどめる\n\n"
        "【ルール】\n"
        "・具体的な練習法・エクササイズ・トレーニング手順は書かない\n"
        "・改善の『キーワード』は出してよいが『やり方』は教えない\n"
        "・各セクション3〜4文。温かく、背中を押すトーンで。\n\n"
        "=== ボイストレーニング専門知識 ===\n\n"
        + knowledge
        + "\n\n必ず以下のJSON形式のみで回答してください（他のテキストは不要）:\n"
        '{"voice_character": "...", "potential": "...", "next_step": "..."}'
    )


def _compute_voice_scores(
    pitch_std: float, mean_hz: float, rms_cv: float,
    brightness_hz: float, harmonic_ratio: float,
    speech_rate: float, rms_trend: float,
) -> dict:
    """各声質指標を1〜10のスコアに変換する（10が最良）"""
    pitch_cv = (pitch_std / mean_hz * 100) if mean_hz > 0 else 0.0

    if   pitch_cv < 2:  s_intonation = 2
    elif pitch_cv < 3:  s_intonation = 3
    elif pitch_cv < 5:  s_intonation = 4
    elif pitch_cv < 8:  s_intonation = 5
    elif pitch_cv < 15: s_intonation = 9
    elif pitch_cv < 25: s_intonation = 7
    else:               s_intonation = 4

    if   rms_cv < 15:   s_dynamics = 3
    elif rms_cv < 25:   s_dynamics = 6
    elif rms_cv < 40:   s_dynamics = 9
    elif rms_cv < 55:   s_dynamics = 8
    else:               s_dynamics = 5

    if   brightness_hz < 500:  s_brightness = 3
    elif brightness_hz < 700:  s_brightness = 5
    elif brightness_hz < 1000: s_brightness = 9
    elif brightness_hz < 1300: s_brightness = 8
    elif brightness_hz < 1800: s_brightness = 6
    else:                      s_brightness = 4

    if   harmonic_ratio < 18: s_resonance = 2
    elif harmonic_ratio < 30: s_resonance = 3
    elif harmonic_ratio < 40: s_resonance = 4
    elif harmonic_ratio < 50: s_resonance = 5
    elif harmonic_ratio < 55: s_resonance = 6
    elif harmonic_ratio < 65: s_resonance = 8
    else:                     s_resonance = 10

    if   speech_rate < 0.5: s_tempo = 3
    elif speech_rate < 0.8: s_tempo = 5
    elif speech_rate < 1.5: s_tempo = 9
    elif speech_rate < 2.0: s_tempo = 7
    elif speech_rate < 2.5: s_tempo = 5
    else:                   s_tempo = 3

    if   rms_trend < -35: s_sustain = 2
    elif rms_trend < -28: s_sustain = 3
    elif rms_trend < -20: s_sustain = 4
    elif rms_trend < 10:  s_sustain = 9
    elif rms_trend < 25:  s_sustain = 7
    else:                 s_sustain = 6

    return {
        "intonation": s_intonation,
        "dynamics":   s_dynamics,
        "brightness": s_brightness,
        "resonance":  s_resonance,
        "tempo":      s_tempo,
        "sustain":    s_sustain,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _knowledge_text
    _knowledge_text = _load_pdf_knowledge()
    pdf_count = len(list(KNOWLEDGE_DIR.glob("*.pdf")))
    print(f"PDF知識読み込み完了: {pdf_count}ファイル / {len(_knowledge_text)}文字")
    print(f"APIキー: {'設定済み' if _API_KEY else '未設定！'}")
    # 初回のライブラリ初期化を利用者の録音ではなく起動時に済ませる。
    sample = np.arange(96000) / 16000
    buffer = io.BytesIO()
    sf.write(buffer, 0.1 * np.sin(2 * np.pi * 150 * sample), 16000, format='WAV')
    buffer.seek(0)
    warmup_file = UploadFile(filename='startup-warmup.wav', file=buffer)
    result = analyze_audio(warmup_file, 'five_preview', DIAGNOSIS_CONFIG['reading']['id'], True)
    if result.status_code != 200:
        raise RuntimeError('Voice analysis startup check failed')
    buffer.close()
    yield


app = FastAPI(title="声診断アプリ", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")
app.mount(TEST_PREFIX + "/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="test-static")
ACCESS_CLOSED = True  # 完成後の新しい期間限定入口を用意するまで診断を停止する。


@app.middleware("http")
async def preparation_gate(request, call_next):
    if ACCESS_CLOSED and not entry_access_allowed(request.url.path, request.method):
        if request.method in ('GET', 'HEAD'):
            from fastapi.responses import HTMLResponse
            return HTMLResponse('<!doctype html><html lang="ja"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>声診断・準備中</title><main style="max-width:600px;margin:15vh auto;padding:24px;font-family:sans-serif">'
                '<h1>声診断は現在準備中です</h1><p>このURLからのご利用は終了しました。</p>'
                '<p>準備が整いましたら、新しいテスト用URLをご案内します。</p></main></html>',
                headers={'Cache-Control': 'no-store', 'Permissions-Policy': 'microphone=()'})
        return JSONResponse({'success': False, 'code': 'test_closed',
                             'error': 'このURLからのご利用は終了しました。新しいテスト用URLのご案内をお待ちください。'},
                            status_code=403, headers={'Cache-Control': 'no-store'})
    return await call_next(request)


@app.middleware("http")
async def microphone_policy(request, call_next):
    response = await call_next(request)
    allowed = not ACCESS_CLOSED or entry_access_allowed(request.url.path, request.method)
    response.headers["Permissions-Policy"] = "microphone=(self)" if allowed else "microphone=()"
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Robots-Tag'] = 'noindex, nofollow'
    return response


@app.get("/help/microphone")
async def microphone_help():
    return FileResponse(Path(__file__).resolve().parent / "static" / "microphone-help.html")


@app.get("/mic-test")
async def microphone_test():
    return FileResponse(Path(__file__).resolve().parent / "static" / "microphone-test.html")


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
    # スコア（1-10）
    score_intonation: int = 5  # 抑揚の幅
    score_dynamics:   int = 5  # 声の強弱
    score_brightness: int = 5  # 声の明るさ
    score_resonance:  int = 5  # 声の響き
    score_tempo:      int = 5  # 話す速さ
    score_sustain:    int = 5  # 声の持続力


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

    score_items = [
        ("抑揚の幅",   req.score_intonation),
        ("声の強弱",   req.score_dynamics),
        ("声の明るさ", req.score_brightness),
        ("声の響き",   req.score_resonance),
        ("話す速さ",   req.score_tempo),
        ("声の持続力", req.score_sustain),
    ]
    high_items = [f"{n}({s}点)" for n, s in score_items if s >= 7]
    low_items  = [f"{n}({s}点)" for n, s in score_items if s <= 4]
    score_guidance = (
        (f"▶ 強み（高スコア）: {', '.join(high_items)}\n" if high_items else "") +
        (f"▶ 課題（低スコア）: {', '.join(low_items)}\n"  if low_items  else "")
    )

    user_prompt = (
        f"以下の音声分析データをもとに、3つのセクションで声診断コメントを生成してください。\n\n"
        f"【基本データ】\n"
        f"- 録音時間: {req.duration}秒\n"
        f"- 最低音: {req.min_note}（{req.min_hz} Hz）\n"
        f"- 最高音: {req.max_note}（{req.max_hz} Hz）\n"
        f"- 平均音程: {req.mean_note}（{req.mean_hz} Hz）\n"
        f"- 音域の幅: 約{semitones}半音\n\n"
        f"【声質スコア（10点満点）】\n"
        + "\n".join(f"- {n}: {s}点" for n, s in score_items) + "\n"
        + score_guidance + "\n"
        f"【声質の詳細特徴】\n"
        f"- 抑揚・音程の揺れ: {pitch_label}\n"
        f"- 声の強弱: {dynamic_label}\n"
        f"- 発声の密度: {voiced_label}\n"
        f"- 声の明るさ・声質: {brightness_label}\n"
        f"- 声の響き・倍音: {harmonic_label}\n"
        f"- 話す速さ・テンポ: {rate_label}\n"
        f"- 声の持続力・傾向: {trend_label}\n\n"
        f"セクション:\n"
        f"1. voice_character：高スコア項目（強み）を中心に、この声の個性・魅力を具体的に描写する。音域データと組み合わせて『自分の声ってそんな特徴があるの！』と気づかせる。褒めて声への関心を高める。\n"
        f"2. potential：低スコア項目（課題）を中心に、改善の入口だけを見せる。『共鳴』『芯』『表情』『息の支え』などのキーワードは使ってよいが、具体的なやり方・練習法は絶対に書かない。「実際の声をお聴きすれば、さらに正確な診断とあなただけのアドバイスをお伝えできます」という前向きな一文で締める。\n"
        f"3. next_step：20分Zoom無料声診断で『あなただけの声の設計図』が見えてくることを伝える。診断を受けることで何がわかるか・どう変わるかを魅力的に描写し、申込みへの一歩を後押しする言葉で締める。人名・固有名詞は使わず『プロの声診断』『専門家』などの表現にとどめる。"
    )

    async def event_stream():
        import re, traceback, asyncio
        max_retries = 2  # 最大2回リトライ（合計3回試みる）

        for attempt in range(max_retries + 1):
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
                return  # 成功

            except Exception as e:
                error_str = str(e)
                is_overloaded = "overloaded" in error_str.lower()

                if is_overloaded and attempt < max_retries:
                    wait_sec = 4 * (attempt + 1)  # 4秒、8秒
                    yield f"data: {json.dumps({'type': 'retrying', 'attempt': attempt + 1, 'wait': wait_sec})}\n\n"
                    await asyncio.sleep(wait_sec)
                    continue

                # リトライ上限 or その他のエラー
                yield f"data: {json.dumps({'type': 'error', 'error': error_str, 'overloaded': is_overloaded})}\n\n"
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
@app.get("/speed-retest")
async def root():
    return FileResponse(str(Path(__file__).resolve().parent / "static" / "index.html"))


@app.get(TEST_PREFIX)
@app.get(TEST_PREFIX + '/')
@app.get(TEST_PREFIX + '/speed-retest')
async def test_root():
    return HTMLResponse(render_entry_html('index.html'))


@app.get(TEST_PREFIX + '/help/microphone')
async def test_microphone_help():
    return HTMLResponse(render_entry_html('microphone-help.html'))


@app.get(TEST_PREFIX + '/mic-test')
async def test_microphone_check():
    return HTMLResponse(render_entry_html('microphone-test.html'))


MAX_AUDIO_BYTES = 8 * 1024 * 1024
MIN_AUDIO_SECONDS = 5.0
MAX_AUDIO_SECONDS = 21.0  # 画面は20秒で停止。端末の停止処理に1秒の余裕を持たせる。


def recording_error(code: str, message: str, status: int = 422):
    return JSONResponse({"success": False, "code": code, "error": message}, status_code=status)


@app.post("/analyze")
def analyze_audio(file: UploadFile = File(...), measurement_mode: str = Form('legacy'),
                  prompt_id: str = Form(''), reading_complete: bool = Form(False)):
    # 同期ルートとして解析をワーカースレッドへ移し、他のHTTP処理を止めない。
    try:
        contents = file.file.read(MAX_AUDIO_BYTES + 1)
        if not contents:
            return recording_error("empty_file", "録音が空です。もう一度、5〜10秒ほど話してください。")
        if len(contents) > MAX_AUDIO_BYTES:
            return recording_error("file_too_large", "録音データが大きすぎます。5〜10秒ほどで録音し直してください。", 413)
        try:
            info = sf.info(io.BytesIO(contents))
        except Exception:
            return recording_error("invalid_audio", "録音を読み取れませんでした。もう一度録音してください。")
        if info.duration < MIN_AUDIO_SECONDS:
            return recording_error("too_short", "録音が短すぎます。5〜10秒ほど話してから止めてください。")
        if info.duration > MAX_AUDIO_SECONDS:
            return recording_error("too_long", "録音が長すぎます。5〜10秒ほどで録音し直してください。")
        if info.channels > 2 or info.samplerate > 192000:
            return recording_error("invalid_audio", "この録音形式は利用できません。ブラウザから録音し直してください。")
        # 16 kHz にダウンサンプリング：pyin の計算量を大幅削減
        audio_data, sr = librosa.load(io.BytesIO(contents), sr=16000, mono=True)
        duration = len(audio_data) / sr
        if not np.all(np.isfinite(audio_data)):
            return recording_error("invalid_audio", "録音を読み取れませんでした。もう一度録音してください。")
        signal_rms = librosa.feature.rms(y=audio_data, hop_length=256)[0]
        # 暫定の入力品質基準。5項目の採点基準とは別に管理する。
        audible = signal_rms >= 10 ** (-55 / 20)
        overall_rms = float(np.sqrt(np.mean(audio_data ** 2)))
        if overall_rms < 10 ** (-55 / 20) or np.sum(audible) * 256 / sr < 0.3:
            return recording_error("silence", "話し声を十分に確認できませんでした。マイクとの距離を確認し、普段の声で5秒以上話してください。")

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
        valid_pitch = np.array([x is not None for x in pitch_hz])
        voiced_frames = valid_pitch & audible[:len(valid_pitch)]
        if np.mean(voiced_frames) < 0.08 or np.sum(voiced_frames) * 256 / sr < 0.2:
            return recording_error("no_voice", "話し声を十分に確認できませんでした。静かな場所で、普段の声で5〜10秒ほど話して録音し直してください。")
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

            voice_scores = _compute_voice_scores(
                pitch_std, mean_hz, rms_cv, brightness_hz,
                harmonic_ratio, speech_rate, rms_trend,
            )

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
                "scores": voice_scores,
            })

        speed_result = None
        if measurement_mode == 'five_preview':
            speed_result = measure_fixed_reading_speed(audio_data, sr, prompt_id, reading_complete,
                                                      DIAGNOSIS_CONFIG['speed_retest'])
        diagnosis = build_diagnosis(audio_data, sr, prompt_id, reading_complete,
            speed_measurement=speed_result) if measurement_mode == 'five_preview' else None
        return JSONResponse({
            "success": True,
            **({"diagnosis": diagnosis} if diagnosis else {}),
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

    except Exception:
        logging.exception("Audio analysis failed")
        return recording_error("analysis_failed", "解析に失敗しました。少し待って、もう一度録音してください。", 500)


@app.post(TEST_PREFIX + '/analyze')
def analyze_test_audio(file: UploadFile = File(...), measurement_mode: str = Form(''),
                       prompt_id: str = Form(''), reading_complete: bool = Form(False)):
    if measurement_mode != 'five_preview':
        return recording_error('invalid_mode', 'この入口では指定の文章による声診断をご利用ください。')
    return analyze_audio(file, measurement_mode, prompt_id, reading_complete)


@app.get(TEST_PREFIX + '/api/diagnosis-config')
@app.get('/api/diagnosis-config')
def diagnosis_config():
    base = Path(__file__).parent / 'knowledge_comments'
    policy = json.loads((base / 'recording_change_policy_v1.json').read_text(encoding='utf-8'))
    entries = json.loads((base / 'approved' / 'recording_change_comments_v1.json').read_text(encoding='utf-8'))['entries']
    totals = json.loads((base / 'approved' / 'high_score_comments_v1.json').read_text(encoding='utf-8'))['entries']
    return {**DIAGNOSIS_CONFIG, 'change_policy': policy,
            'change_comments': [{k: e[k] for k in ('id', 'section', 'text', 'text_without_comparison_audio') if k in e} for e in entries],
            'total_comments': [{k: e[k] for k in ('id', 'text', 'score_min_inclusive', 'score_max_exclusive') if k in e} for e in totals]}
