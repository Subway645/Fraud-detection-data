# 净来电 — 诈骗电话识别 · 数据管道

> 成员A（算法数据）的 TTS 语音合成与数据集构建工具链。
> 基于 Microsoft Edge-TTS 生成诈骗/正常通话语音，支持音频增强、质量检查、数据划分。

## 项目结构

```
TTS/
├── config.py          # 全局路径与划分配置
├── main.py            # 批量 TTS 语音生成
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

## 各脚本说明

### 1. `main.py` — 批量 TTS 语音生成

读取 `text_data/` 下的话术 CSV 表，调用 Edge-TTS 将文本合成为 MP3 音频。

- **输入**：`fraud_utterances.csv` / `ad_utterances.csv`（需包含 `text`、`type`、`voice` 三列）
- **输出**：`audio_data/{fraud_audio,ad_audio}/` 下的 `.mp3` 文件
- **并发控制**：`SEMAPHORE_LIMIT = 10`，单条失败自动重试 3 次

**用法**：

```python
# 在 main.py 中修改 SELECTED 列表：
SELECTED = ["fraud", "ad"]   # 同时生成两类
SELECTED = ["ad"]            # 仅生成广告语音
SELECTED = ["fraud"]         # 仅生成诈骗语音

# 然后运行：
python main.py
```

**输出文件命名**：`{序号:03d}_{类别}.mp3`，如 `001_冒充公检法.mp3`、`ad_001_保险推销.mp3`

---

### 2. `post_process.py` — 音频增强

对已生成的原始音频施加数据增强，增加模型训练时的鲁棒性。

支持以下增强方式（通过 CSV 中的 `augment_*` 列控制）：

| 列名 | 含义 | 取值范围 |
|------|------|----------|
| `augment_speed` | 变速 | 0.5~2.0，1.0=不变 |
| `augment_noise` | 加噪 | 0~1，数值越大噪声越强 |
| `augment_reverb` | 混响 | 0~1 |
| `augment_pitch` | 音调偏移 | 正值升调，负值降调 |
| `augment_low_quality` | 低音质模拟 | 0~1，通过降采样实现 |
| `augment_volume` | 音量调整 | 0~2，1.0=不变 |

**用法**：

```python
CATEGORY = "both"   # "fraud" / "ad" / "both"
python post_process.py
```

**输出**：`audio_data/{类别}/processed/` 下以 `_processed.wav` 结尾的文件。

---

### 3. `fix_missing.py` — 批量修复损坏/缺失音频

扫描所有预期输出文件，检查是否存在且大小 ≥ 5KB（小于此阈值视为损坏），对缺失或损坏的条目重新调用 TTS 生成。

**用法**：

```python
CATEGORY = "both"        # "fraud" / "ad" / "both"
MIN_FILE_SIZE = 5000     # 字节，小于此值视为损坏
python fix_missing.py
```

适合在 `main.py` 执行后运行，确保所有音频完整生成。

---

### 4. `fix_one.py` — 修复单条音频

按索引号重新生成某一条特定音频，适合快速修复某条音质不理想的语音。

**用法**：

```python
CATEGORY = "ad"   # "fraud" 或 "ad"
INDEX = 1         # 第几条（从 1 开始）
python fix_one.py
```

---

### 5. `split_data.py` — 数据集划分

按分层采样将话术 CSV 划分为训练集、验证集、测试集。

- **划分比例**：7:1.5:1.5（在 `config.py` 中可调）
- **分层依据**：`label` 列（保证各类别在各集合中比例一致）
- **固定随机种子**：`random_state=42`，保证可复现

**用法**：

```python
# 直接运行，自动处理 fraud_utterances.csv 和 ad_utterances.csv
python split_data.py
```

**输出**：

```
text_data/splits/
├── fraud/
│   ├── fraud_utterances_train.csv
│   ├── fraud_utterances_val.csv
│   └── fraud_utterances_test.csv
└── ad/
    ├── ad_utterances_train.csv
    ├── ad_utterances_val.csv
    └── ad_utterances_test.csv
```

---

### `config.py` — 全局配置

```python
BASE_DIR = r"C:\Users\Subway\Desktop\诈骗电话识别"

TEXT_DIR   # 话术 CSV 所在目录 (text_data/)
AUDIO_DIR  # 音频输出根目录 (audio_data/)
FRAUD_DIR  # 诈骗音频子目录
AD_DIR     # 广告/正常音频子目录

SPLIT_OUTPUT_DIR  # 划分输出目录
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
```

迁移到其他机器时，只需修改 `BASE_DIR` 即可。

---

## 完整工作流

按 7.21–7.26 的任务顺序，典型使用流程如下：

```bash
# Step 1 — 准备话术 CSV（fraud_utterances.csv / ad_utterances.csv，含 text, type, voice 列）

# Step 2 — 批量生成原始音频
python main.py

# Step 3 — 检查并补全缺失/损坏文件
python fix_missing.py
# 个别不满意可单独修复：
python fix_one.py

# Step 4 — 音频增强（可选，需先在 CSV 中填写 augment_* 列）
python post_process.py

# Step 5 — 数据集划分
python split_data.py
```

---

## 话术 CSV 格式

| 列名 | 说明 | 示例 |
|------|------|------|
| `text` | 话术文本 | 您好，这里是公安局，您的身份证涉嫌洗钱... |
| `type` | 诈骗/广告类别标签 | 冒充公检法、贷款诈骗、保险推销 |
| `voice` | Edge-TTS 音色 | zh-CN-XiaoxiaoNeural、zh-CN-YunxiNeural |
| `label` | 最终标签（用于划分分层） | 1=诈骗, 0=正常 |

增强模式下还需 `augment_speed`、`augment_noise`、`augment_reverb` 等列。
