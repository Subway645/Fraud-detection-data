# 净来电 — 诈骗电话识别 · 数据管道

> 成员A（算法数据）的 TTS 语音合成与数据集构建工具链。
> 基于 Microsoft Edge-TTS 生成诈骗/广告/正常通话语音，支持音频增强、质量检查、数据划分。

## 项目结构

```
TTS/
├── config.py          # 全局路径与划分配置
├── main.py            # 批量 TTS 语音生成（三类：诈骗/广告/正常）
├── post_process.py    # 音频增强（变速/加噪/混响/音调/低音质/音量）
├── fix_missing.py     # 扫描损坏/缺失音频并重新生成
├── fix_one.py         # 修复单条音频
├── split_data.py      # 分层训练/验证/测试集划分
└── README.md
```

## 环境依赖

```
pip install edge-tts pandas librosa soundfile numpy scikit-learn
```

## 数据集总览

| 类别 | CSV 文件 | 条数 | label | type 种类 |
|------|----------|------|-------|----------|
| 诈骗 | fraud_utterances.csv | 368 | fraud | 28 种 |
| 广告 | ad_utterances.csv | 270 | ad | 14 种 |
| 正常 | normal_utterances.csv | 470 | normal | 2 种 |
| **合计** | | **1108** | | |

### CSV 列说明

| 列名 | 说明 | 示例 |
|------|------|------|
| `text` | 话术文本 | 喂妈，我下班了，晚上回来吃饭 |
| `label` | 类别标签 | fraud / ad / normal |
| `type` | 子类别 | 裸聊敲诈 / 家人问候 / 保险推销 |
| `voice` | Edge-TTS 音色 | zh-CN-XiaoxiaoNeural |
| `augment_reverb` | 混响强度 | 0.00 ~ 0.12 |
| `augment_noise` | 噪声强度 | 0.000 ~ 0.012 |
| `augment_speed` | 语速倍率 | 1.00 ~ 1.35 |
| `augment_low_quality` | 低音质模拟 | 0.00 ~ 0.80 |
| `augment_volume` | 音量系数 | 0.75 ~ 1.55 |
| `augment_pitch` | 音调偏移（半音） | -2 ~ +2 |

---

## 各脚本说明

### 1. `main.py` — 批量 TTS 语音生成

读取 `text_data/` 下的 CSV，调用 Edge-TTS 将文本合成为 MP3 音频。

```python
# 配置区（修改 SELECTED 切换类别）
SELECTED = ["fraud"]                    # 仅诈骗
SELECTED = ["ad"]                       # 仅广告
SELECTED = ["normal"]                   # 仅正常
SELECTED = ["fraud", "ad", "normal"]    # 全量

# 运行
python main.py
```

- **输入**：`fraud_utterances.csv` / `ad_utterances.csv` / `normal_utterances.csv`
- **输出目录**：`audio_data/fraud_audio/` / `ad_audio/` / `normal_audio/`
- **文件命名**：`{prefix}{序号:03d}_{类型}.mp3`，如 `001_裸聊敲诈.mp3`、`normal_001_家人问候.mp3`
- **并发**：`SEMAPHORE_LIMIT = 10`，单条失败自动重试 3 次

---

### 2. `post_process.py` — 音频增强

对原始音频施加数据增强，增加模型鲁棒性。

| 参数 | 含义 | 典型范围 |
|------|------|----------|
| `augment_speed` | 变速 | 1.00~1.35 |
| `augment_noise` | 加噪 | 0.000~0.012 |
| `augment_reverb` | 混响 | 0.00~0.12 |
| `augment_pitch` | 音调偏移 | -2~+2 |
| `augment_low_quality` | 低音质模拟 | 0.00~0.80 |
| `augment_volume` | 音量调整 | 0.75~1.55 |

```python
# 配置区
CATEGORY = "fraud"    # 单类
CATEGORY = "normal"   # 正常
CATEGORY = "all"      # 全量三类

python post_process.py
```

输出到 `audio_data/{类别}/processed/`，文件名以 `_processed.wav` 结尾。

---

### 3. `fix_missing.py` — 批量修复损坏/缺失音频

扫描所有预期输出文件，检查是否存在且 ≥ 5KB，对缺失或损坏条目重新 TTS。

```python
CATEGORY = "all"          # "fraud" / "ad" / "normal" / "all"
MIN_FILE_SIZE = 5000      # 小于此值视为损坏

python fix_missing.py
```

---

### 4. `fix_one.py` — 修复单条音频

按索引号重新生成某一条特定音频。

```python
CATEGORY = "normal"   # "fraud" / "ad" / "normal"
INDEX = 7             # 第几条（从 1 开始）

python fix_one.py
```

---

### 5. `split_data.py` — 数据集划分

按 label 分层采样，划分训练/验证/测试集。

- 比例：7:1.5:1.5（`config.py` 可调）
- 随机种子：`random_state=42`

```bash
python split_data.py
```

输出结构：

```
text_data/splits/
├── fraud/
│   ├── fraud_utterances_train.csv
│   ├── fraud_utterances_val.csv
│   └── fraud_utterances_test.csv
├── ad/
│   ├── ad_utterances_train.csv
│   ├── ad_utterances_val.csv
│   └── ad_utterances_test.csv
└── normal/
    ├── normal_utterances_train.csv
    ├── normal_utterances_val.csv
    └── normal_utterances_test.csv
```

---

### `config.py` — 全局配置

```python
BASE_DIR = r"C:\Users\Subway\Desktop\诈骗电话识别"

TEXT_DIR      # text_data/
AUDIO_DIR     # audio_data/
FRAUD_DIR     # audio_data/fraud_audio/
AD_DIR        # audio_data/ad_audio/
NORMAL_DIR    # audio_data/normal_audio/

SPLIT_OUTPUT_DIR  # text_data/splits/
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
```

迁移时只需修改 `BASE_DIR`。

---

## 完整工作流

```bash
# 1. 准备 CSV（已就绪）
#    text_data/fraud_utterances.csv   368条诈骗
#    text_data/ad_utterances.csv      270条广告
#    text_data/normal_utterances.csv  470条正常

# 2. 批量生成原始音频（按需选择）
python main.py    # 改 SELECTED = ["fraud", "ad", "normal"]

# 3. 检查并补全缺失/损坏文件
python fix_missing.py    # CATEGORY = "all"

# 4. 音频增强
python post_process.py   # CATEGORY = "all"

# 5. 数据集划分
python split_data.py
```


