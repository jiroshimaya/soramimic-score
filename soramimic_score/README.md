# 開発者向けガイド

音声解析、歌詞と音符の対応付け、JSONの保存・読込をPythonから利用できます。
インストールと音源解析コマンドは[README](../README.md)を参照してください。

## 標準の音声解析

`analyze_audio(path, model_config=config)`は次の処理を行います。

1. Whisperで歌詞の文と行ごとの時刻を認識します。
2. MeCabとUniDic Liteで日本語の発音形を生成します。
3. ReazonSpeechのかなCTC出力を使い、選んだ読みに沿って発音単位の時刻を推定します。
4. SheetSage2でメロディの音高と開始・終了時刻を推定します。
5. 発音と音符を対応付け、`ScoreDocument`を返します。

正式歌詞を`lyrics=("一行目", "二行目")`として渡すとWhisperを省略します。
歌詞の文字列は保持し、読みと時刻を別に推定します。行時刻がない正式歌詞は音源全体に
整列します。対応言語は日本語です。辞書で読めない語やCTCで扱えない文字はエラーになります。
ボーカル分離や音声による読み候補の選び直しは行いません。

音声モデルは呼び出す段階で読み込み、推論後に解放します。CPUとCUDAを選択できます。
SheetSage2は指定ディレクトリの付属コードを実行します。重み・コードとも信頼できる
配布元から用意してください。モデルの利用条件は[外部ライブラリ・モデルの表記](../THIRD_PARTY_NOTICES.md)
を参照してください。

## モデルの差し替え

独自の推論処理を使う場合は、4つの関数を`AudioAdapters`に指定します。

```python
from soramimic_score import AudioAdapters, analyze_audio, dump

adapters = AudioAdapters(
    lyric_recognizer=recognize_lyrics,
    reading_selector=select_readings,
    mora_aligner=align_moras,
    melody_transcriber=transcribe_melody,
)
score = analyze_audio("song.wav", adapters)
dump(score, "work/song.score.json")
```

上記の関数は利用側で定義します。モデル設定と`adapters`の同時指定はできません。

| 関数 | 入力 | 戻り値の各要素 |
| --- | --- | --- |
| 歌詞認識 | 音源の`Path` | `LyricLine`：文字列、開始・終了時刻 |
| 読み選択 | 音源、歌詞行 | `ReadingSelection`：カナ、出典、スコア |
| 発音時刻推定 | 音源、歌詞行、読み | `AlignedMora`：行番号、発音単位番号、カナ、時刻、信頼度 |
| メロディ推定 | 音源 | `MelodyNote`：時刻、MIDI音高、出典、任意の信頼度 |

時刻は入力音源の先頭からの秒数です。歌詞行、発音、メロディはそれぞれ時系列順で、
同種の区間が重ならないように渡します。発音列は選択した読みのモーラ列と一致させます。
音高の信頼度が不明なら`None`を使います。辞書の読みスコアやCTCスコアは、
認識の正しさを保証する確率ではありません。

## 観測データからの変換

`build_audio_observations(lines, readings, aligned_moras, melody_notes)`はモデルの結果を
`IntermediateRepresentation`へまとめます。これは歌詞・読み・発音時刻・音符候補・
根拠を保持する、バージョン付きの入力形式です。

```python
from soramimic_score import compile_score, dump, load

score = compile_score(observations)
dump(score, "work/song.score.json")
same_score = load("work/song.score.json")
```

`compile_score`には、対応付け前の観測を渡します。対応付け済みの場合は
`from_linked_observations`を使うと、再計算せず歌唱用の音符列を組み立てられます。
`build_known_lyrics_document`は指定歌詞にも自動認識で選択済みの歌詞にも利用できる
補助関数で、それ自体は音声認識を行いません。

## 対応付けの詳細

標準の対応付けは、発音時刻とメロディノートを使って、各音符で何を発音するかを決めます。
同じ発音に属する同音高の断片をまとめ、音高が変わっても発音が続く場合は継続音にします。
対応が見つからない歌詞は削除せず、未解決として保持します。

細かな設定は`soramimic_score.pipeline.run_stage3_document`で指定できます。
歌詞行の時刻には`line_windows_by_utterance`を使います。反復発声の回数を音響根拠から
補う場合は、`vocalization_reattacks_by_utterance`に`VocalizationReattack`を渡します。
標準の音源解析は、この反復補完用の追加認識を実行しません。

比較実験には`align_correspondence`と`run_correspondence_document`も使えます。
これらは発音と音符のまとまりをsemi-Markov方式で探索する別の対応付けアルゴリズムです。
標準の音源解析や`compile_score`では呼び出しません。

## 出力JSON

`ScoreDocument`は元の観測、対応付け、根拠、歌唱用の音符列を一つに保存します。
結合・分割した音符は元の候補を参照する根拠を持ち、保存した結果から追跡できます。

- `canonical`：歌詞と発音単位のID
- `performed`：観測状態と対応付け
- `synthesis_plan`：音高・時刻・発音を持つ歌唱用の音符列

形式名は`format: "soramimic-score"`、形式の版は整数の`schema_version`です。
パッケージのバージョンとは独立しています。

CLIで観測JSONを変換する場合は次のように指定します。

```sh
soramimic-score --input work/observations.json --output work/song.score.json
```

対応付け済みの入力には`--linked-input`を付けます。
