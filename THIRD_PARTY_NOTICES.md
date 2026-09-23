# 外部ライブラリとモデル

本リポジトリのコードはMITライセンスです。外部ライブラリ、辞書、モデルの重み、
モデル付属コードには、それぞれの配布元のライセンスが適用されます。
本リポジトリのMITライセンスでこれらの条件を置き換えることはありません。

## 音声モデル

| 用途 | モデル | ライセンス・配布元 |
| --- | --- | --- |
| 歌詞認識 | Whisper / faster-whisper変換モデル | [MIT](https://github.com/openai/whisper/blob/main/LICENSE)、[変換モデル](https://huggingface.co/Systran/faster-whisper-large-v3) |
| 発音時刻 | ReazonSpeech japanese-wav2vec2-base-rs35kh | [Apache-2.0](https://huggingface.co/reazon-research/japanese-wav2vec2-base-rs35kh) |
| 音高推定 | SheetSage2 | [CC BY-NC 4.0のモデル表記](https://huggingface.co/m-a-p/SheetSage2)、[LICENSE](https://huggingface.co/m-a-p/SheetSage2/blob/main/LICENSE) |
| SheetSage2の基盤 | MERT-v2-FullSong | [CC BY-NC 4.0](https://huggingface.co/m-a-p/MERT-v2-FullSong) |

SheetSage2とMERT2の重みは同梱・自動取得しません。利用条件を確認したローカル配置を
指定してください。付属Pythonコードも実行されるため、信頼できる配布物を使用してください。
モデルを再配布する場合は、元の著作権表示、ライセンス、帰属表示を保持してください。
SheetSage2側の[外部素材に関する表記](https://huggingface.co/m-a-p/SheetSage2/blob/main/THIRD_PARTY_NOTICES.md)
も適用されます。

## ライブラリと辞書

直接依存に加え、インストールされる間接依存にも各配布物の条件が適用されます。
再配布時は使用バージョンに同梱されたライセンスと著作権表示を保持してください。

- 基本処理: [jasyllablesep](https://pypi.org/project/jasyllablesep/)
- 音声認識: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)、[CTranslate2](https://github.com/OpenNMT/CTranslate2)
- 推論: [PyTorch](https://github.com/pytorch/pytorch)、[torchaudio](https://github.com/pytorch/audio)、[Transformers](https://github.com/huggingface/transformers)
- 読み生成: [mecab-python3](https://github.com/SamuraiT/mecab-python3)、[UniDic Lite](https://github.com/polm/unidic-lite)
- 音声・数値処理: [librosa](https://github.com/librosa/librosa)、[SoundFile](https://github.com/bastibe/python-soundfile)、[NumPy](https://numpy.org/)
- モデル付属処理: [mir_eval](https://github.com/craffel/mir_eval)、[pretty_midi](https://github.com/craffel/pretty-midi)

## 生成物

本コードのMITライセンスは、入力楽曲・歌詞や生成物の権利を許諾するものではありません。
入力素材の権利と、使用したモデルの利用条件を確認してください。
生成したJSONやMIDIに、このライブラリが一律のライセンスを付与することはありません。
