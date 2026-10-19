# Physical AI 応用編1 — Genesis World × Colab × リスク適応巡回（2026-10-16）

**作品案**：教師なし設備異常検知と動的巡回計画を組み合わせた、自作差動二輪ロボットの3D工場点検。

**状況**：Colab Notebook・Pythonコード・URDF生成・手順書を提供。**ColabのGenesis GPU物理／EGL映像化はこの環境では実行できておらず、成功未確認**。AIプランナー・URDF生成はローカル単体試験済み（`pytest` 3件成功）。失敗時の診断手順を明記。

## 最初に開くファイル

| 順序 | ファイル | 機能・必要リソース |
|---|---|---|
| 1 | `docs/Genesis_Colab_セットアップ手順書.md` | 準備・依存関係・トラブル対処 |
| 2 | `notebooks/00_Genesis_Colab_Setup_Smoke.ipynb` | インストール、EGL/カメラ/MP4確認（GPU推奨） |
| 3 | `notebooks/01_AI_Patrol_Benchmark.ipynb` | 合成センサ、Isolation Forest、A*、2D運動学ベースライン比較（CPU可） |
| 4 | `notebooks/02_Genesis_3D_Rover_Mission.ipynb` | 自作差動二輪URDF、**関節速度入力と物理位置フィードバック**、AIによる目標変更、3D動画 |
| 5 | `notebooks/03_Report_and_Submission.ipynb` | 指標の監査、レポート下書き、動画尺確認 |
| 参考 | `docs/先行研究_技術ブログ_設計レビュー.md` | 講義資料との対応、一次文献/ブログ、採否・限界 |

Notebook 01/02は**それぞれ必要なPythonソースをNotebookセルに内包**しているため、個別にColabへアップロードして実行できます。Notebook 03の評価データは01/02でダウンロードした`outputs`フォルダのファイルをアップロードします。

## 提出物と表現上の注意

- `outputs/PAI_final_task_3D.mp4`: Genesisのカメラ映像にAIリスクと目標点を重ねた3D物理走行の提出候補。02実行後に生成。
- `outputs/genesis_physical_trace.csv` / `outputs/genesis_physical_metrics.json`: Genesisの剛体位置から計算した値。**ジオフェンス近接は実際の接触力ではない**。
- `outputs/planner_only_metrics.csv`: 01の2D運動学比較。**Genesis物理の性能として報告しない**。
- `outputs/report_draft_genesis.md`: 03で生成するPDF原稿の下書き。
- `assets/pai_rover.urdf`: 02内で生成される自作ロボット定義。必要なら出力済みURDFをリポジトリに加える。

使用する合成センサデータは工場実測でもデジタルツイン実測でもありません。真の故障情報はAIの異常スコアに直接渡さず、評価用として保持しています。

## テスト

```bash
python -m pip install numpy pandas scikit-learn pytest
python -m pytest tests/ -q
```

`genesis-world`はColab上のNotebook 00/02で別途セットアップ。2026年10月16日時点の公開安定版 **1.4.4**（2026-10-11公開）を明示的に指定。最新追従よりもバージョン記録を重視しています。

## 課題期限

2026年11月2日（月）10:00 JST。OmniCampus PDF、GitHub Public、YouTube MP4 (1〜3分) のすべてが必要。氏名・所属・メールアドレスを成果物に含めないこと。

## 参考

Genesis公式: https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html  
Rendering: https://genesis-world.readthedocs.io/en/latest/user_guide/rendering/index.html  
Control: https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/control_your_robot.html  
Repository: https://github.com/Genesis-Embodied-AI/genesis-world

---

## 追加：Hugging Face ML実画像学習版（2026-10-16）

過去の2D工場巡回・IsolationForest版は保持しつつ、新たにHugging Faceの道路異常写真・MobileNetV3-Small分類・任意PPO速度学習・Genesis物理巡回を追加した。

| 順 | Notebook | 説明 |
|---|---|---|
| 0 | `notebooks/00_Genesis_Colab_Setup_Smoke.ipynb` | Genesisヘッドレスレンダリングの動作確認 |
| 1 | `notebooks/04_HF_Road_Issues_MobileNetV3_Train.ipynb` | HFトークン（Colab Secret）、データ取得、転移学習、検証、出力`road_ml_assets.zip` |
| 2 (任意) | `notebooks/05_PPO_Risk_Aware_Speed_Training.ipynb` | 別の1D速度代理環境でPPO学習、出力`ppo_speed.zip` |
| 3 | `notebooks/06_Genesis_HF_Vision_Integrated_Mission.ipynb` | 外部仮想道路カメラ画像に基づくGenesis 3D移動ロボットの物理走行、90s MP4 |

詳しい認証と実行方法：`docs/HF_機械学習_Genesis_実行手順書.md`
文献と実装／未実装の明確な区分：`docs/先行研究_技術資料_機械学習改善レビュー.md`

**シミュレーションの完全な画像閉ループではない**：HF画像は外部仮想点検カメラの入力であり、GenesisからレンダリングしたオンボードRGBフレームではない。実画像の分類とGenesis物理制御を結合するプロトタイプである。ColabGPU/GenesisEGLの実実行は本環境から未検証。動画・データ・重み生成を虚偽表示しない。


## Colab Secrets（この版の設定）

- Notebook `04_HF_Road_Issues_MobileNetV3_Train.ipynb` は、Colabのシークレット名 **`HFH_READ`** から Hugging Face のReadトークンを取得します。
- Colab左側「シークレット」で `HFH_READ` を登録し、**ノートブックからのアクセス**をONにしてください。
- トークンの値はコード、Notebook出力、GitHubに書かないでください。
- 公開データセットを利用する場合、未認証でも動く構成になっています。ただし登録したトークンを使うにはアクセス権限をONにしてください。
