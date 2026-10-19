# Genesis World × Google Colab セットアップ／実行手順書

**対象**：Physical AI応用編1・2026年11月2日10:00 JST提出課題  
**作成日**：2026年10月16日  
**ターゲット**：Colab Linux / Python 3.10–3.13 / Genesis World **1.4.4** / 可能ならNVIDIA T4以上  
**実行検証の境界**：依存関係および公式APIをWeb資料で確認し、Python/URDFのローカル検査に成功。**本作成環境ではGenesisとColab GPU/OpenGLの実機組合せを起動していない**。したがってNotebook 00の実行結果で可否を確定する。

## 1. 設計原則

本構成はColab内部で**3D環境（Genesis）・ロボット制御・合成センサAI推論・録画を完結**させる。ローカルにROS 2/Gazebo/Unity/Blenderをインストールしない。コードを公開GitHubに置くため、機密情報、氏名、メール、APIキー、ColabのDrive内部パスを残さない。

技術基盤は4層に分離する。

1. **知覚**：仮想設備の温度・振動・電流・音圧の合成観測を生成。モデルは正常サンプルだけを学習したIsolation Forest。故障ラベルを推論器に渡さない。
2. **判断・計画**：設備ごとのリスクと移動距離を統合して目標を選択。A*グリッド経路を計算し、遠隔アラーム発生時に再計画。
3. **作用・物理**：Genesis 1.4.4上のURDF差動二輪ロボットの左右車輪関節を速度制御。剛体の位置・向きを取得し、その現在位置から追従制御。
4. **検証・運用**：3D実走行ログ、別個の2Dアルゴリズム比較、乱数シード/ライブラリバージョン/動画を保存。

**講義との対応**：第1回「モデル単体でなくシステム全体を評価・ログ管理・Sim-to-Realギャップに注意」、第6回「OSS・再現可能なGitHub公開」を反映。

## 2. Colabでの作業順（最短経路）

1. ZIPを展開し、`notebooks/00_Genesis_Colab_Setup_Smoke.ipynb`をGoogle Colabにアップロードする。
2. メニューで `ランタイム` → `ランタイムのタイプを変更` → `T4 GPU`（または使用可能なGPU）を選択。メニュー表記は変更され得る。
3. 00の全セルを上から実行。最初のセルでGPU、Python、PyTorch、CUDAを確認する。`nvidia-smi`が失敗する場合、Colab GPU割当を確認。
4. 次のセルでAPTのEGL/OpenGL/OSMesa、FFmpegを導入し、`pip install genesis-world==1.4.4` などを実行。大容量パッケージのため、時間がかかることがある。
5. **`import genesis`より前に**`PYOPENGL_PLATFORM=egl`を指定。既にPyOpenGLをimportしている場合の設定変更は同一カーネルでは効かないことがある。
6. PNG画像と2秒MP4が保存されたら、`outputs/environment_manifest.json` と併せてダウンロードする。ここが**必須の最初のゲート**。
7. 新しいColabランタイムで01をアップロード。スモークテスト、12乱数シード×2方式比較、CSV/PNG、参考2D動画を生成する。
8. 新しいColabランタイムで02をアップロード。最初に2秒間の自作URDF/関節制御スモーク、次に90秒3D走行（10fpsで記録）を実行する。
9. 02が成功したら`outputs/PAI_final_task_3D.mp4`、`genesis_physical_trace.csv`、`genesis_physical_metrics.json`、`genesis_visits.csv`、`genesis_events.json`、`run_manifest.json`を保存する。
10. 01の`planner_only_metrics.csv`と02の`genesis_physical_metrics.json`等を03の`outputs/`にアップロードし、説明資料の下書きと動画尺を検証する。
11. 提出用PDFへ図表・シミュレーション映像のキャプチャを追加。コードをGitHub Publicに登録し、動画はYouTubeへアップ。OmniCampusの各提出欄にURL等を記入する。

> **Notebook 00/01/02は別々に実行可能**。Colabはセッションが変わると `/content/outputs` のデータが消えるため、各段階で必ずダウンロードする。Driveの利用は任意であり、セットアップの必須条件ではない。

## 3. 描画エラー時の診断フロー

| 状況 | 推定原因・確認 | 対応 |
|---|---|---|
| `GenesisException: No display detected` | `show_viewer=True` が使われている | `show_viewer=False`、Camera `GUI=False`にする |
| `eglInitialize` / `eglQueryString` / `GL` 関連例外 | EGLランタイムとPyOpenGLの不整合 | まず`ldconfig -p | grep EGL`、再起動して `PYOPENGL_PLATFORM=osmesa` を試す。変更はimportより前 |
| GPUは認識するが `cam.render()` が極端に遅い | メサCPU描画へフォールバック等 | `nvidia-smi`とEGL情報確認、解像度を320×240に下げる。GPU物理≠GPU描画 |
| `ModuleNotFoundError: genesis` | `%pip`実行前／異なるカーネル | インストールセルから再実行 |
| `Python incompatible` | Colab Pythonが3.14など | 3.10〜3.13を利用できる環境に変更（Colab提供バージョンに依存） |
| URDFの関節が見つからない | 解析されたジョイント名が違う | `rover.get_joint`の名称確認、`pai_rover.urdf`と一致させる |
| 車輪が回るのに前進しない | 摩擦・接地・車輪回転符号・トルク | 02のスモークで基部位置を観察。`wheel_sign`（±1）、`kv`、`force_range`、接地初期位置を調整 |
| ロボットが進路から逸脱 | 簡易PID/車輪滑り/障害物干渉 | 速度を0.3m/sへ下げる、`max_omega`を調整、2秒検証→短いデモから |
| `ffprobe` / MP4なし | renderer/encoder出力問題 | 00のカメラ録画テストから。FFmpegインストール確認 |
| 時間・メモリ不足 | Colabランタイムのリソース変動 | 解像度480×320、fps5〜10、`duration_s`は60〜180sを維持して負荷削減 |

**重要**：OSMesaはCPU描画を使う場合があり、CUDAで高速物理演算できてもMP4のレンダリングが遅い可能性がある。2025年の個人検証には「Colabでcam.renderが非常に遅かった」報告があるが、その環境は旧Genesisであり、2026年版の成否を断定できない。

## 4. 生成ソースと記録対象

- `pai_patrol.py`：過去2D Notebookを継承した正常学習Isolation Forest、合成センサ、A*、巡回ポリシー、ペア比較。これは**物理シミュレーションを含まない**。
- `genesis_mission.py`：Genesisで3D設備・障害物・自作差動二輪URDFを生成、車輪速度指令、位置取得、オンラインターゲット更新と3Dログ。
- `pai_visuals.py`：2Dアルゴリズム比較の参考動画と図表。Genesis実写扱いは禁止。
- `outputs/genesis_adaptive.mp4`：Genesisカメラの素映像。
- `outputs/PAI_final_task_3D.mp4`：上記映像にログから生成したリスク・目標設備のラベルを重畳。**合成センサであることを画面明示**。
- `outputs/genesis_physical_metrics.json`：走行距離はGenesisの剛体位置差分。`geofence_proximity_steps_proxy`は境界への接近観測で**接触センサでない**。
- `outputs/planner_only_metrics.csv`：2D運動学ベースライン。3D物理測定と混ぜない。

### 2D比較と3D実行を分ける理由

同じ合成センサと計画ロジックを共有していても、2Dでは位置積分が理想的で、3Dではタイヤと床の接触、滑り、質量、慣性が作用する。したがって、2Dで観測されたリコール改善を「Genesisでも改善」と記すのは誤り。3D動画では**AIが目標を変更し、物理ロボットが追従したか**を見せ、3Dでの厳密な比較を行いたければ同じ乱数シードで`policy='nearest'`と`policy='adaptive'`を個別に再走行する（`02`を別ランタイムで2回実行）。

## 5. 動画とレポートの構成例

- 0:00〜0:15　目的・設備異常の説明（工場の3D全景）
- 0:15〜0:40　走行・設備の点検
- 0:40〜1:15　遠隔異常スコアが上昇、目標変更（画面テロップ）
- 1:15〜1:30　実験の限界と次の改善点

動画は**Genesisの3D録画が主**。01の2D動画を本編と誤認させない。90秒に満たない時は出力尺と実際のシミュレーション時間を整合。時間短縮編集で倍速を使った際は動画/概要欄に倍率を表示する。

## 6. 課題提出の確認

- PDF：`説明資料_PAI最終課題_omnicampusアカウント名.pdf`。技術構成、手法、AIの必然性、3D実験結果、比較の限界、工夫を明記。
- GitHub URL：**public**。ノートブック4本、src、docs、requirements相当、README、無害な検証ログを登録（必要なら大きなMP4は除く）。公開前に実データ・個人情報・秘密値をチェック。
- YouTube：`PAI最終課題_omnicampusアカウント名`、1〜3分、最低限**限定公開**、「子ども向け」「年齢制限」は「いいえ」、公開設定を判定完了まで維持。
- OmniCampus：PDF、GitHub URL、YouTube URL。**2026年11月2日（月）AM10:00**。

## 7. APIとトラブルシュートの一次資料

- Genesis World installation: https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html
- Genesis World rendering/headless/video: https://genesis-world.readthedocs.io/en/latest/user_guide/rendering/index.html
- Genesis World control and joint index: https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/control_your_robot.html
- Genesis World URDF: https://genesis-world.readthedocs.io/en/latest/api_reference/engine/entity/morph/file_morph/urdf.html
- Genesis 1.4.4 package publication: https://pypi.org/project/genesis-world/1.4.4/
- EGL import failure #3129: https://github.com/Genesis-Embodied-AI/genesis-world/issues/3129
- Colab/GPU記事（2025・旧バージョンの事例）: https://tototototototo7.hatenablog.com/entry/2025/genesis_test_google_colab
- DataCamp 2025 implementation: https://www.datacamp.com/tutorial/genesis-physics-engine-tutorial

最新ドキュメントURLは改訂で変わる可能性がある。`run_manifest.json`のバージョンと実行ログを残す。
