# CODEX_RULES.md

## プロジェクト概要

このプロジェクトは「voice-app-v2」です。

目的は、現行の声診断アプリとは独立した次世代版の開発です。

現行運用中の voice-app とは完全に分離して作業してください。

---

# 編集可能範囲

編集可能

* voice-app-v2 配下のファイル
* voice-app-v2 配下の新規作成ファイル
* voice-app-v2 配下の設定ファイル
* voice-app-v2 配下のGit管理対象

---

# 参照可能範囲

参照のみ許可

* C:\Users\climb\Desktop\voice-app
* C:\Users\climb\Desktop\voice-knowledge
* knowledge
* knowledge_archive

参照は許可します。

内容の分析・要約・設計への反映は可能です。

---

# 絶対禁止事項

以下はユーザーから明示的な指示がない限り禁止です。

## voice-app

禁止対象

C:\Users\climb\Desktop\voice-app

禁止内容

* ファイル編集
* ファイル削除
* ファイル移動
* ファイル名変更
* Git Commit
* Git Push
* Pull Request作成
* Render設定変更
* Renderデプロイ
* 環境変数変更
* リポジトリ設定変更

voice-app は現行運用中の本番資産です。

閲覧のみ許可します。

---

# デプロイルール

Codex がデプロイを実行してはいけません。

禁止

* Render Deploy
* Render Service作成
* 本番環境更新
* 本番環境設定変更

デプロイ作業はユーザー確認後のみ実施します。

---

# Gitルール

編集可能

* voice-app-v2 のブランチ作成
* voice-app-v2 のコミット

禁止

* voice-app リポジトリへのコミット
* voice-app リポジトリへのPush
* voice-app リポジトリへの変更

---

# 設計方針

voice-app-v2 は以下を目的とする。

* 録音開始率向上
* 診断完了率向上
* LINE登録率向上
* Zoom予約率向上

現行版の完全コピーではなく、新しい集客導線の実験環境として扱う。

---

# 知識利用方針

voice-knowledge は共通資産である。

以下を優先的に参照する。

* 診断理論
* 声質評価基準
* コメントテンプレート
* 改善アドバイス
* Zoom診断フレームワーク

実装判断は玉井メソッドを優先する。

---

# 判断に迷った場合

不明点がある場合は、

「編集せず提案のみ」

を優先すること。

勝手に変更しない。
