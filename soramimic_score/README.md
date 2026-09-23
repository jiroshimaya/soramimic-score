# Soramimic Score Stage 3コア

`soramimic_score`は、音声モデルに依存しない歌詞・音符対応付けの本番コアです。
選択済みの歌詞・読みと、独立に推定されたメロディのノート候補を対応付け、元の情報へ
戻れる歌唱計画へ変換します。

音声モデルの実行方法は`AudioAdapters`で差し替えます。モデル自体と運用上の判断policyは、
このライブラリの利用側が与えます。

- Demucsによる音源分離
- 未知歌詞に対する通常Whisperの表層認識
- 日本語の読み候補生成とKanaWhisperによる再順位付け
- ReazonSpeechかなCTCによるモーラ時刻の整列
- SheetSage2によるメロディ推定

このパッケージはモデルを同梱しませんが、音源から各adapterを呼び出し、結果を検証して
正本JSONへ変換する実行経路を提供します。既知歌詞と自動認識した歌詞は、表層と読みを
選んだ後、同じStage 3経路へ入ります。

## 音源からの実行

`analyze_audio`は、次の4種類のadapterを順番に呼び出します。

1. 歌詞行の認識（正式歌詞を渡した場合は省略）
2. 各行の読みの選択
3. モーラ時刻の整列
4. メロディノートの推定

```python
from soramimic_score import AudioAdapters, analyze_audio, dump

adapters = AudioAdapters(
    lyric_recognizer=recognize_lyrics,
    reading_selector=select_readings,
    mora_aligner=align_moras,
    melody_transcriber=transcribe_melody,
)
score = analyze_audio("song.wav", adapters)
dump(score, "song.score.json")
```

正式歌詞がある場合は`lyrics`へ行単位で渡します。この場合、歌詞認識adapterは呼ばれません。

```python
score = analyze_audio("song.wav", adapters, lyrics=("一行目", "二行目"))
```

adapter間の値には`LyricLine`、`ReadingSelection`、`AlignedMora`、`MelodyNote`を使います。
時刻・順序・かな・MIDI音高・信頼度はStage 3へ入る前に検証されます。モデル固有の生出力や
checkpointは、この境界の外に置きます。

## 入力契約

`IntermediateRepresentation`がversioned JSONの入力境界です。新しいStage 3入力は、
次の情報を持ちます。

- 正規の発話、選択した読み、モーラ、歌唱単位
- 呼び出し側が与えた音響根拠とモーラ時刻
- メロディadapterが出力した時系列順の`NoteCandidate`
- 対応付け済みlinkをまだ含まないこと

`build_known_lyrics_document`は、歌詞とモーラ側の文書を組み立てる補助関数です。
歴史的な名前とは異なり、上流で選択済みの自動認識表層にも使用できます。この関数は
認識処理を行いません。

モデル固有の生出力は各adapter内に留めます。この境界を越えるのは、正規化した観測、
信頼度、ID、出典情報だけです。

## 本番runner

```python
from soramimic_score import VocalizationReattack
from soramimic_score.pipeline import run_stage3_document

run = run_stage3_document(
    observation_document,
    line_windows_by_utterance={"u0": (1.2, 4.8)},  # 任意のWhisper生区間
)

linked_document = run.document
realization = run.realization
note_run_decision = run.note_run
```

自動認識の呼び出し側は、Whisperが丸めた反復発声の回数を、独立した音響passから
復元できます。

```python
run = run_stage3_document(
    observation_document,
    line_windows_by_utterance={"u0": (1.2, 4.8)},
    vocalization_reattacks_by_utterance={
        "u0": (
            VocalizationReattack(1.25, 1.31, 0.8, "raw-kana-ctc"),
            VocalizationReattack(1.62, 1.68, 0.7, "raw-kana-ctc"),
        ),
    },
)
```

Whisperは反復するモーラの種類だけを供給し、復元後の回数は供給しません。呼び出し側が
渡す再発音時刻から、固定上限を設けずに回数を決めます。重なるSheetSage2ノートは
音高と長さを決め、歌唱されるノートとして必ず保持します。再発音の間にあるノートは、
新しい子音や破棄される末尾ノートではなく、継続音になります。推定した各反復には、
音響sourceと信頼度を明示的な根拠として残します。再発音時刻を渡さない場合、通常歌詞と
既知歌詞の処理は変わりません。

`run_stage3_document`は変更不能な観測文書を検証し、隣接する発話のCTC範囲の中点で
ノートを分割します。その後、ノートを保持するdecoderを一度実行し、再生可能なlinkと
派生ノート候補を実体化して、すべてのlinkをまとめてcompileします。既存linkの黙示的な
再利用は行わず、入力に含まれていれば拒否します。

scoreが使用するのは、歌詞の開始時刻（通常は生のモーラCTC）、選択した歌詞segmentの
identity、音高付きSheetSage2区間です。反復発声の復元時だけ、呼び出し側が与えた再発音
時刻を使います。自動認識の呼び出し側は、各発話のWhisper生区間も渡せます。割り当てた
ノートが区間外から始まり、モーラCTC開始だけが区間内にある場合、decoderは小さな二次の
行所有costを加えます。区間内から始まったノートは、末尾が境界を越えても全体を保持します。
CTC開始も境界を越える場合は、信頼度の閾値なしで越境を支持します。

scoreは参照/XFノート、F0、ノート信頼度、歌唱単位区間を入力に使いません。同じ音高の
隣接fragmentをまとめるのは、同じ選択済みsyllable内だけです。別のCTC syllableは境界を
維持し、音高変化は明示的な継続slotになります。音高のない候補は明示的な`rest` linkとして
残します。

## 対応付けと歌唱実現

対応付け済み出力は、すべての生SheetSage2候補を変更せず保持します。結合または分割した
最終区間は、元候補を示す`note-run-derivation`根拠を持つ追加の`NoteCandidate`として
保存します。そのため、保存した文書からoptimizerを再実行せず、`compile_realization`で
同じ結果を再現できます。

`compile_realization`は、次の三つのlayerを分けて公開します。

- `canonical`: 完全な表層テキストと安定したモーラID
- `performed`: 観測状態と選択した対応付け
- `synthesis_plan`: 具体的なノートslotへ割り当てた発音

対応付けが欠けても、正規歌詞を削除しません。実際の歌唱省略には、正の根拠を持つ
明示的な`PerformanceOmission`が必要です。

## 別の対応付けmodel

`align_correspondence`と`run_correspondence_document`は、比較用として以前の
semi-Markov対応付けmodelを保持しています。本番の`run_stage3_document`および
公開している音源解析経路からは呼び出しません。

## 正本JSONとCLI

保存時の正本は、一つの`ScoreDocument`です。対応付け済み観測とcompile済みscoreを
まとめるため、exporterやapplicationは一つの入力だけを扱えば済みます。

```python
from soramimic_score import compile_score, dump, load

score = compile_score(observation_document)
dump(score, "song.score.json")
same_score = load("song.score.json")
```

新しい観測文書をdecodeし、一つのscore文書として保存します。

```sh
python -m soramimic_score \
  --input work/observations.json \
  --output work/song.score.json
```

すでにlink済みの観測文書を、再decodeせずcompileすることもできます。

```sh
python -m soramimic_score \
  --input work/correspondence.json \
  --linked-input \
  --output work/song.score.json
```

このパッケージはモデルに依存せず、音源、checkpoint、楽曲データを含みません。
