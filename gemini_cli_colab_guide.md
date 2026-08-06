# Google Colab上でGoogle DriveのファイルをGemini CLIで操作する手順

Google Colab Pro環境でGoogle Driveをマウントし、Drive上のプログラムやCSVファイルをGemini CLIで直接操作・分析するための手順書です。

## 1. Google Driveをマウントする
Colabの新しいセルに以下のコードを貼り付けて実行します。
アクセス許可を求めるポップアップが表示されたら、許可してください。

```python
from google.colab import drive
drive.mount('/content/drive')
```
※これで、Google Driveの中身が `/content/drive/MyDrive/` 以下に展開されます。

## 2. Gemini CLIをインストールする
Colabはセッションが切れると初期化されるため、起動のたびにCLIのインストールが必要です。
次のセルで以下を実行します。

```bash
!npm install -g @google/gemini-cli
```

## 3. 対象のフォルダに移動する
プログラムやCSVが保存されているGoogle Drive内のフォルダへ移動します。
（パスはご自身の環境に合わせて適宜変更してください）

```bash
%cd /content/drive/MyDrive/あなたのフォルダ名
```

## 4. Gemini CLIでファイルを参照して実行する
Colab上からGemini CLIを実行する際は、先頭に `!` をつけます。
`@` を使ってDrive上のファイルを直接プロンプトに読み込ませることができます。

### CSVデータを分析させる例
Kaggleなどのデータセットを分析する際も、ファイル名を指定するだけで読み込ませることができます。

```bash
!gemini -p "このデータセットの特徴量と目的変数の関係について傾向を分析して @train.csv"
```

### プログラムのコードレビューや修正をさせる例
実装中のソースコードを指定して、エラー原因の特定やリファクタリングを依頼できます。

```bash
!gemini -p "このJavaプログラムで発生している例外の原因を特定して修正案を出して @Main.java"
```

## 💡 Google Colabで扱う際のアドバイス
* **大容量のCSVファイルについて**:
  Gemini CLIを使ってプロンプトに直接巨大なCSV（数十MB以上など）を読み込ませると、トークン数（一度に処理できる文章量）の制限に引っかかる可能性があります。データが大きい場合は、Colab上でPandasなどのPythonライブラリを使ってデータを前処理（要約・抽出）した上で、その結果をGeminiに渡すアプローチがおすすめです。
* **Colabでの対話モード**:
  Colabのセル内で `!gemini` を実行して対話モードに入ると、入力欄の挙動がターミナルと異なり操作しづらい場合があります。Colab上では `-p` （プロンプト）オプションを使って、**1回の実行ごとに完結させる使い方（ワンショット実行）**が最もスムーズです。
