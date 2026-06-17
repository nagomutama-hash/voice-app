# CLAUDE.md

## プロジェクト概要

このプロジェクトは「voice-app」です。

現行運用中の本番声診断アプリです。

次世代版の開発は voice-app-v2 で行います。このプロジェクトと完全に分離して作業してください。

---

## 編集可能範囲

編集可能

* voice-app 配下のファイル
* voice-app 配下の新規作成ファイル
* voice-app 配下の設定ファイル
* voice-app 配下のGit管理対象

---

## 参照可能範囲

参照のみ許可

* C:\Users\climb\Desktop\voice-app-v2
* C:\Users\climb\Desktop\voice-knowledge
* knowledge
* knowledge_archive

参照は許可します。

内容の分析・要約・設計への反映は可能です。

---

## 絶対禁止事項

以下はユーザーから明示的な指示がない限り禁止です。

### voice-app-v2

禁止対象

C:\Users\climb\Desktop\voice-app-v2

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

voice-app-v2 は次世代版の開発資産です。閲覧のみ許可します。

---

## デプロイルール

Claude がデプロイを実行してはいけません。

禁止

* Render Deploy
* Render Service作成
* 本番環境更新
* 本番環境設定変更

デプロイ作業はユーザー確認後のみ実施します。

---

## Gitルール

編集可能

* voice-app のブランチ作成
* voice-app のコミット

禁止

* voice-app-v2 リポジトリへのコミット
* voice-app-v2 リポジトリへのPush
* voice-app-v2 リポジトリへの変更

---

## 設計方針

voice-app は現行の本番環境です。

* 既存の録音・診断・LINE登録フローを維持する
* 不具合修正・UI改善は慎重に行う
* 大きな構造変更は voice-app-v2 で実験してから反映する

---

## 知識利用方針

voice-knowledge は共通資産です。

以下を優先的に参照します。

* 診断理論
* 声質評価基準
* コメントテンプレート
* 改善アドバイス
* Zoom診断フレームワーク

実装判断は玉井メソッドを優先します。

---

## 判断に迷った場合

不明点がある場合は、

「編集せず提案のみ」

を優先すること。

勝手に変更しない。
