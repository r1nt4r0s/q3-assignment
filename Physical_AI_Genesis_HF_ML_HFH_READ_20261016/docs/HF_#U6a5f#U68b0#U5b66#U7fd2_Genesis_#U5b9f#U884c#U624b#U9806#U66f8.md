# Physical AI 2026 応用編1：Hugging Face × MobileNetV3 × PPO × Genesis 実行手順

最終更新：2026-10-16。課題締切：2026-11-02（月）10:00（JST）。

## 0. コンセプトとデータの由来

公道の巡回監視に似せた**仮想インフラ点検施設**に移動ローバーを配置する。外部の「仮想道路カメラ」はHugging Faceの道路写真を入力し、MobileNetV3-Smallが問題の**カテゴリ**を判定する。カテゴリの予測分布を明示的な仮説的リスク重みに写像して、3Dロボットの訪問先・接近時速度を変える。ここで実際に車輪を動かし、位置を観測するのはGenesisの物理エンジン。

**制約を明示する**：HF写真そのものはGenesisカメラがレンダリングしたものではない。HF画像を仮想固定カメラから配信する「外部センサ結合」構成である。画像識別結果は損傷の重大度ではなく、種別の予測。速度安全性は物理環境・実機で保証していない。

## 1. 事前準備

1. Colab（https://colab.research.google.com/）で作業。04, 05 はGPU優先、06はGPU推奨。
2. Hugging Face（https://huggingface.co/settings/tokens）でreadまたはfine-grained readのUser Access Tokenを作成する。書き込み権限は不要。**公開データなので未認証でも原則利用可能**。
3. Colab左サイドバーの鍵アイコン「シークレット」→ `HFH_READ` を登録 → 当該Notebookのアクセスを許可。**Tokenの文字列はNotebookのどこにも貼り付けない**。実行ログに出さない。
4. GPU利用時は「ランタイム」→「ランタイムのタイプを変更」→ T4 GPU等を選択。ColabのGPU種類・利用時間は保証されない。
5. 以前の`00_Genesis_Colab_Setup_Smoke.ipynb`を先に動かし、画像とMP4が出るか確認。失敗したら `docs/Genesis_Colab_セットアップ手順書.md` を参照。

## 2. Notebook実行順

| 順番 | ファイル | 役割 | 成果物 |
|---|---|---|---|
| 00 | `00_Genesis_Colab_Setup_Smoke.ipynb` | Genesis導入・EGL/オフスクリーン描画 | smoke MP4 |
| 04 | `04_HF_Road_Issues_MobileNetV3_Train.ipynb` | データセット取得・品質確認・MobileNetV3分類器学習・モデル評価 | `road_ml_assets.zip`, Macro-F1、混同行列 |
| 05 | `05_PPO_Risk_Aware_Speed_Training.ipynb` | 速度制御PPOの追加発展（**1D簡易環境**） | `ppo_speed.zip`、比較CSV |
| 06 | `06_Genesis_HF_Vision_Integrated_Mission.ipynb` | HF分類＋動的巡回計画＋Genesis実車輪関節制御 | `outputs/PAI_final_task_HF_Genesis.mp4`、物理走行CSV |
| 03 | `03_Report_and_Submission.ipynb` | 既存のレポート生成手順を参考に統合結果をレポートへ | PDFドラフト |

**05は任意**。学習が間に合わない場合でも、04と06だけで学習モデルをGenesis動作へ接続できる。

### Notebook 04

- `HFH_READ` Secret読み込み。秘密情報はNotebook出力に記録しない。
- `datasets.load_dataset(REPO_ID, split='train', revision=commit_sha, token=HFH_READ)`で固定revisionを取得。
- classごとに最大180件、合計最大1,260件（データセットのクラス数7）を抽出し、train/val/testに層化分割。
- ImageNet事前学習済み `torchvision.models.mobilenet_v3_small` の特徴抽出器を固定し、分類器だけ3 epoch更新。
- Validationで温度スケーリングを調整し、testでMacro-F1・混同行列・分類レポート。
- testの異なる6カテゴリから画像を選び、モデル重みとともに`road_ml_assets.zip`へ保存。
- `train/test`に実質的に同一の画像の拡張コピーが混在する可能性があるため、結果を一般化性能の証拠と過大評価しない。

### Notebook 05（オプション）

- `gymnasium`を使う独自の1D走行代理環境でPPOを学習。
- 速度報酬・リスク加重のオーバースピード罰則・加速ペナルティを用いる。
- 固定速度とルールベース速度の両方をペアシードで比較する。
- これは**GenesisによるPPO学習ではない**。06ではGenesisの物理走行に再適用し、引き続きヒューリスティックな最大速度上限を適用する。

### Notebook 06

- Notebook 04で得た `road_ml_assets.zip` をアップロード。
- 05の`ppo_speed.zip` を使う場合だけ、同ファイルをColabへアップロードして`USE_PPO=True`にする。
- `RUN_3D=True`なら90秒分の物理シーンを実行し、走行記録と動画を保存。
- `RUN_BASELINE=True`にするとNearest方式でも90秒走行（Colab計算時間増大）。
- 提出動画の右上には外部仮想カメラのHF画像が挿入され、**HF VIRTUAL FEED**と表示される。
- `CAP_PROP_FRAME_COUNT/FPS`で動画の60〜180秒制約を再検査。

## 3. 比較実験とレポート

- **写真の分類**：test Macro-F1、混同行列、クラス別Recall／Precision、温度スケーリングの有無。
- **巡回行動**：Nearest vs AI-category riskの点検到着時刻、Genesis剛体から算出した移動距離、訪問数。ロボットの実姿勢・位置を使う。
- **速度**：Constant vs safe-rule vs PPO（05内の代理環境）; 実際のGenesis速度追従は06で別途評価。異なる実験を同じ母集団として混ぜない。
- **制約**：HF写真とGenesis描画ピクセルの視覚的ドメインは異なる。画像分類のカテゴリから重大度を割り当てる写像は人手設定。動画の時点で実機安全保証・GPS自律走行は実現していない。

## 4. 主要なトラブル

| エラー | 対応 |
|---|---|
| `401/403` | gatedの場合はWeb画面でアクセス許可申請済みか確認。ColabでSecretのNotebook accessをON。publicはSecret不要で試せる。 |
| HFデータが取得できない | 同じrepo/revisionをWebブラウザで確認。容量とデータセット作成者の更新も確認。 `load_dataset` を再試行。 |
| `label`列がない | 作者が形式を変更した可能性。`raw.features`を確認してコードの構造適合を行う。自動で誤ラベルを付けて学習しない。 |
| CUDA OOM | `BATCH_SIZE=16`、`MAX_PER_CLASS=80`に変更。MobileNetV3-Smallを継続使用。 |
| Genesis EGL・OpenGL | Notebook 00を優先。Colabのドライバやレンダラー適合は非保証。`PYOPENGL_PLATFORM`はimport前に設定。 |
| wheel does not move | 過去セットアップ手順のURDF軸、`wheel_sign`、PID gainを確認。 |
| MP4尺エラー | 実MP4のFPSとフレーム数を確認。FFmpeg再エンコードで1〜3分に調整、必要なら倍速表記。 |
| PPO fails/unconvincing | 05を使用せず、ルールベースの速度制限で06へ。成果が良くない時はそのまま比較記載。 |

## 5. 提出

- YouTube：1〜3分MP4、タイトル`PAI最終課題_omnicampusアカウント名`、限定公開以上、子ども向けNo、年齢制限No。
- GitHub：**public**、Notebook・ソース・ライセンスへのリンク・実行手順・コード。APIトークン、個人名、所属、メールなどは除外。
- OmniCampus：説明PDF `説明資料_PAI最終課題_omnicampusアカウント名.pdf`、public GitHub URL、YouTube URL。
- 最終締切：**2026年11月2日10:00（日本時間）**。

## 6. 参考URL

- HF Access Tokens: https://huggingface.co/docs/hub/security-tokens
- データセット: https://huggingface.co/datasets/Programmer-RD-AI/road-issues-detection-dataset
- HF Image datasets: https://huggingface.co/docs/datasets/image_load
- MobileNetV3: https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html
- PPO stable-baselines3: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- Genesis rendering: https://genesis-world.readthedocs.io/en/latest/user_guide/rendering/index.html
