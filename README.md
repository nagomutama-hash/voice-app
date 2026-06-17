# voice-app-v2

現行の声診断アプリとは独立して検証する、改善版の声診断 Web アプリです。
玉井智による「信頼される声診断サービス」として見せることを目的に、録音開始率・診断完了率・LINE登録率・Zoom無料声診断予約率の改善を検証します。

---

## 必要な環境

- Python 3.9 以上
- pip

---

## インストール手順

### 1. 仮想環境の作成（推奨）

```bash
python -m venv venv
```

```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 2. ライブラリのインストール

```bash
pip install -r requirements.txt
```

| ライブラリ | 用途 |
|---|---|
| `fastapi` | Web フレームワーク |
| `uvicorn` | ASGI サーバー |
| `python-multipart` | ファイルアップロード処理 |
| `numpy` | 数値計算 |
| `librosa` | 音声解析・音程検出 (pYIN アルゴリズム) |
| `soundfile` | 音声ファイルの読み込み |

> **注意:** `librosa` のインストールには依存関係が多く、数分かかる場合があります。

---

## アプリの起動

```bash
python -m uvicorn main:app --reload
```

> **注意:** `uvicorn` コマンドが PATH に入っていない場合は `python -m uvicorn` を使ってください（Windows でよく発生します）。

起動後、ブラウザで以下の URL を開いてください。

```
http://localhost:8000
```

---

## 使い方

1. **「無料で声を診断する」** をクリックしてマイクの使用を許可する
2. 画面の例文に沿って7秒ほど話す
3. **「録音を止める」** をクリック
4. 録音した音声が自動的に解析され、グラフが表示される
5. 診断結果で声の強み・伸びしろを確認する

---

## ディレクトリ構成

```
voice-app-v2/
├── main.py            # FastAPI バックエンド（音声解析 API）
├── requirements.txt   # 依存ライブラリ
├── render.yaml        # Render 別Service用の設定
├── README.md          # このファイル
└── static/
    └── index.html     # フロントエンド UI
```

---

## トラブルシューティング

### `librosa` のインストールに失敗する場合

Windows では Microsoft C++ ビルドツールが必要になることがあります。  
以下を試してください。

```bash
pip install --upgrade pip setuptools wheel
pip install librosa
```

### マイクにアクセスできない場合

- ブラウザの設定でマイクの使用を許可してください。
- `localhost` (HTTP) では通常マイクが使用できます。別のホストから HTTPS なしでアクセスしている場合は制限されます。

### 音程グラフに何も表示されない場合

- 無音や雑音のみの録音では音程を検出できません。
- はっきりとした声（母音など）を 1 秒以上録音してください。
