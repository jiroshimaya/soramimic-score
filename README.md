# Soramimic Score

歌っている音源から、**歌詞つきの音符データ**を作るためのPythonライブラリです。

歌詞、音の高さ、歌い始めと歌い終わりの時刻を一つのJSONにまとめます。このJSONを
もとに、カラオケ表示、XF MIDI、MusicXMLなどを作れるようにすることが目標です。

## 処理の流れ

```mermaid
flowchart TD
    A["歌唱音源"] --> B["音声を読み込み"]
    B --> V["ボーカルを分離<br/>Demucs"]
    B --> C{"正式な歌詞がある?"}
    C -- ある --> D["正式な歌詞をそのまま使用"]
    D -. 削除・補完を選択 .-> A1["音声認識と照合し<br/>行の省略・繰り返し・追加を反映"]
    C -- ない --> E["歌詞を自動認識<br/>例: Whisper"]

    D --> F["読み候補を生成<br/>soramimic-yomi + UniDic Lite"]
    A1 --> F
    E --> F
    F --> R["音声と照らして読み候補を選択<br/>KanaWhisper"]
    B --> R
    V --> R
    R --> G["歌詞を発音単位に分け<br/>開始・終了時刻を推定<br/>ReazonSpeech"]
    V --> G

    B --> H["メロディの音高と<br/>開始・終了時刻を推定<br/>例: SheetSage2"]
    G --> I["歌詞の発音単位と<br/>メロディの音符を対応付け"]
    H --> I

    I --> J["歌詞・読み・時刻・音高・<br/>信頼度と判断根拠をJSONに保存"]
    J -.-> K["カラオケ表示"]
    J -.-> L["XF MIDI"]
    J -.-> M["MusicXML"]
    J -.-> N["字幕"]
```

歌詞側の処理とメロディ側の処理は別々に行い、あとで対応付けます。正式な歌詞が
ある場合は、既定ではその歌詞を変更せず使います。必要なら、音源に合わせて
行を削除・補完するオプションも選べます。JSONを保存の中心にして、
各出力形式はJSONから作ります。点線の出力は今後追加する予定です。

読みは`soramimic-yomi`で生成し、UniDic Liteで別の候補を補います。
原音と分離したボーカルを比較して候補を選び、判断が曖昧なときは最初の読みを維持します。
発音時刻はボーカルから、歌詞本文とメロディは原音から推定します。

## 現在の状態

日本語の歌唱音源から歌詞・読み・時刻・音高を推定し、JSONに保存できます。
音声用の依存ライブラリとモデルの準備が必要です。認識結果には誤りが含まれるため、
利用前に確認してください。

歌詞認識や音高推定などのAIモデル本体は、用途に合わせて差し替えられるように
分離しています。モデルの重みは同梱しません。XF MIDI・MusicXML・字幕への
書き出しは今後追加する予定です。

## 現在の使い方

ソースを取得したディレクトリで音声用の依存をインストールします。Python 3.11を推奨します。
`soramimic-yomi`をGitから取得するため、Gitも必要です。

```sh
pip install '.[audio]'
```

[SheetSage2](https://huggingface.co/m-a-p/SheetSage2)と
[MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong)は、利用条件を確認して
ローカルに用意してください。以下のパスは配置先の例です。

```sh
soramimic-score analyze song.wav \
  --sheetsage-model models/SheetSage2 \
  --sheetsage-base models/MERT-v2-FullSong \
  --output work/song.score.json
```

CPUで実行します。GPUを使う場合は`--device cuda`を付けます。
Whisper・Demucs・KanaWhisper・ReazonSpeechは初回使用時に取得します。取得済みのモデルだけを使う場合は
`--local-files-only`を付けてください。正式歌詞は`--lyrics lyrics.txt`で指定できます
（UTF-8、1行1フレーズ）。

### 入力歌詞を音源に合わせて削除・補完する

`--lyrics lyrics.txt --adjust-lyrics`を付けると、音声認識の結果と照合して、
歌われていない行を外し、繰り返しや不足する行を補います。指定しなければ
入力歌詞は変更しません。Pythonでは`analyze_audio(..., lyrics=lines, adjust_lyrics=True)`です。

補正は**行単位**です。対応する行は入力した表記のまま使い、行途中の言葉は
切り貼りしません。認識側の改行が異なる場合は、前後最大4行をまとめて照合します。
一行も対応が取れない場合は、入力を認識結果で置き換えずエラーにします。
認識ミスによって誤った削除・追加が起こり得るため、結果を確認してください。
元の入力、認識結果、採用した行、削除・補完の判断はJSONの`lyric-adjustment`根拠に残します。

### 音声モデルの設定

ボーカル分離と音声による読み選択は標準で有効です。不要な場合は、
`--no-vocal-separation`（原音で時刻を推定）や`--dictionary-readings`（音声比較をせず読みを生成）を
指定できます。分離済みの音声は一時ファイルとして扱い、解析終了時に削除します。
`--dictionary-readings`の場合も`soramimic-yomi`を使います。
Demucsの取得済み重みを指定する場合は、`--demucs-checkpoint models/955717e8-8726e21a.th`を
付けます。指定しなければPyTorchのモデルキャッシュを使います。

Pythonからは次のように実行します。

```python
from pathlib import Path
from soramimic_score import ModelConfig, analyze_audio, dump

config = ModelConfig(
    sheetsage_model=Path("models/SheetSage2"),
    sheetsage_base=Path("models/MERT-v2-FullSong"),
)
score = analyze_audio("song.wav", model_config=config)
dump(score, "work/song.score.json")
```

認識済みの観測データがある場合は、モデルを実行せず変換できます。

```python
from soramimic_score import compile_score, dump

score = compile_score(observations)
dump(score, "song.score.json")
```

CLIからも同じ変換を実行できます。

```sh
soramimic-score --input observations.json --output song.score.json
```

入力JSONの形式や対応付けの詳細は、
[開発者向け説明](soramimic_score/README.md)を参照してください。

このリポジトリには、音源、完全な歌詞、MIDI、モデルの重み、ユーザーの
投稿データ、評価用正解データを含めません。

## ライセンス

本リポジトリのコードは[MIT](LICENSE)です。依存ライブラリ・辞書・モデルには
それぞれのライセンスが適用されます。標準構成のSheetSage2とMERT2には非商用条件が
あります。[外部ライブラリ・モデルの利用条件](THIRD_PARTY_NOTICES.md)を確認してください。
