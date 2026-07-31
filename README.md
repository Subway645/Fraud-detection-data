# 净来电 — 诈骗电话识别 · 数据管道

> 成员A（算法数据）的 TTS 语音合成与数据集构建工具链。
> 基于 Microsoft Edge-TTS 生成诈骗/广告/正常通话语音，支持音频增强、质量检查、数据划分。

## 项目结构

```
诈骗电话识别/
├── README.md                              ← 当前文件
├── HANDOFF.md                             ← 交接文档（先读那个）
│
├── TTS/                                   # 语音合成工具链
│   ├── config.py                          # 全局路径
│   ├── main.py                            # 批量 TTS 生成
│   ├── post_process.py                    # 音频增强
│   ├── fix_missing.py                     # 批量修复损坏音频
│   ├── fix_one.py                         # 修复单条
│   ├── split_data.py                      # 训练/验证/测试划分
│   └── README.md                          # TTS 子文档
│
├── text_data/                             # 文本数据
│   ├── fraud_utterances.csv               # 诈骗 368 条
│   ├── ad_utterances.csv                  # 广告 270 条
│   ├── normal_utterances.csv              # 正常 470 条
│   ├── pattern_library/                   # 话术模板库
│   │   ├── fraud_patterns.json           # 诈骗 v1.3：28种 · 41模板
│   │   ├── ad_patterns.json              # 广告 v1.3：14种 · 28模板
│   │   └── normal_patterns.json          # 正常 v1.1
│   ├── knowledge_graph/                   # 知识图谱
│   │   ├── entities.csv                  # 实体词典 1089 行
│   │   └── relations.csv                 # 关系表 1248 行
│   └── splits/                            # 训练/验证/测试划分
│       ├── fraud/
│       ├── ad/
│       └── normal/
│
└── audio_data/                            # 生成的语音文件
    ├── fraud_audio/ + processed/
    ├── ad_audio/ + processed/
    └── normal_audio/ + processed/
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

## 话术模式库（pattern_library/）

每种诈骗/广告类型有结构化模板 + 三档权重关键词（strong/medium/weak）。

版本 v1.3（2026.7.31）。三种权重说明：

| 权重 | 含义 | 匹配规则 |
|------|------|---------|
| strong | 几乎只在诈骗/广告中出现 | 单次命中即报警 |
| medium | 诈骗常见但日常也可能出现 | ≥2 个同时命中 |
| weak | 日常高频词 | 不加分，做平滑参考 |

v1.3 统计：

| 库 | Strong | Medium | Weak | 总计 |
|----|--------|--------|------|------|
| 诈骗 | 143 | 27 | 212 | 382 |
| 广告 | 36 | 33 | 84 | 153 |

所有 42 种类型至少 1 个 strong。

### JSON 结构

```json
{
  "metadata": { "version": "1.3", "groups": { "权威压迫型": [...], ... } },
  "patterns": {
    "冒充公检法人员诈骗": {
      "group": "权威压迫型",
      "count": 16,
      "templates": [{
        "pattern": "[机构名] + [涉嫌罪名] + [配合要求]",
        "slots": {
          "机构名": ["公安局", "法院"],
          "涉嫌罪名": ["洗钱", "贩毒"],
          "配合要求": ["转账到安全账户", "提供验证码"]
        }
      }],
      "keywords": [
        {"word": "安全账户", "weight": "strong"},
        {"word": "拘留", "weight": "strong"}
      ],
      "confusable_with": ["虚假征信诈骗"]
    }
  }
}
```

---

## 知识图谱（knowledge_graph/）

从 pattern JSON 导出两张表，不用 Neo4j。给匹配算法和模型联动直接查。

**实体 types:**

| 类型 | 数量 | 含义 |
|------|------|------|
| ScamType | 28 | 诈骗类型 |
| AdType | 14 | 广告类型 |
| ScamGroup / AdGroup | 4 + 2 | 大组（权威压迫型等） |
| Keyword | 77 | ≤2字词，精确匹配 |
| Phrase | 374 | 2-5字短语，子串匹配 |
| Utterance | 343 | 完整话术，整句匹配 |
| Brand | 155 | 机构/品牌名 |
| Action | 92 | 操作指令 |

子类型按 slot 语义 + 长度判定，匹配算法按子类型用不同策略。

**四种关系:**

| 关系 | 方向 | 含义 | 数量 |
|------|------|------|------|
| has_entity | 类型 → Entity | 该类型的模板里有这个词 | 922 |
| indicates | Entity → 类型 | 听到这个词警惕该类型（仅 strong+medium） | 236 |
| belongs_to_group | 类型 → Group | 所属大组 | 42 |
| confusable_with | 类型 ↔ 类型 | 容易互相混淆（双向） | 48 边 / 24 对 |

查法：

```python
# 听到一个词 → 哪种诈骗？
relations[(relations["relation"]=="indicates") & (relations["head_name"]=="安全账户")]

# 一种诈骗 → 会说哪些话？
relations[(relations["relation"]=="has_entity") & (relations["head_name"]=="冒充公检法人员诈骗")]

# 两个类型之间容易混淆吗？
relations[(relations["relation"]=="confusable_with") & (relations["head_name"].isin(["冒充公检法", "虚假征信"]))]
```

---

## 各脚本说明

### 1. `main.py` — 批量 TTS 语音生成

读取 CSV，调用 Edge-TTS 合成 MP3 音频。

```python
# 修改 SELECTED 切换类别
SELECTED = ["fraud"]                    # 仅诈骗
SELECTED = ["ad"]                       # 仅广告
SELECTED = ["normal"]                   # 仅正常
SELECTED = ["fraud", "ad", "normal"]    # 全量

python main.py
```

- **输入**：`fraud_utterances.csv` / `ad_utterances.csv` / `normal_utterances.csv`
- **输出目录**：`audio_data/fraud_audio/` / `ad_audio/` / `normal_audio/`
- **文件命名**：`{prefix}{序号:03d}_{类型}.mp3`，如 `001_裸聊敲诈.mp3`
- **并发**：`SEMAPHORE_LIMIT = 10`，失败自动重试 3 次

---

### 2. `post_process.py` — 音频增强

| 参数 | 含义 | 典型范围 |
|------|------|----------|
| `augment_speed` | 变速 | 1.00~1.35 |
| `augment_noise` | 加噪 | 0.000~0.012 |
| `augment_reverb` | 混响 | 0.00~0.12 |
| `augment_pitch` | 音调偏移 | -2~+2 |
| `augment_low_quality` | 低音质模拟 | 0.00~0.80 |
| `augment_volume` | 音量调整 | 0.75~1.55 |

```python
CATEGORY = "all"      # 或 "fraud" / "ad" / "normal"
python post_process.py
```

输出到 `audio_data/{类别}/processed/`。

---

### 3. `fix_missing.py` — 批量修复损坏/缺失音频

```python
CATEGORY = "all"
python fix_missing.py
```

---

### 4. `fix_one.py` — 修复单条音频

```python
CATEGORY = "normal"
INDEX = 7
python fix_one.py
```

---

### 5. `split_data.py` — 数据集划分

按 label 分层采样，比例 7:1.5:1.5，随机种子 42。

```bash
python split_data.py
```

输出：`text_data/splits/{fraud,ad,normal}/` 下各有 train/val/test 三个 CSV。

---

### `config.py` — 全局配置

```python
BASE_DIR = r"C:\Users\Subway\Desktop\诈骗电话识别"
TRAIN_RATIO = 0.7 / VAL_RATIO = 0.15 / TEST_RATIO = 0.15
```

---

## 完整工作流

```bash
# 1. CSV 已就绪（fraud/ad/normal_utterances.csv）

# 2. 批量生成原始音频
python main.py    # 改 SELECTED = ["fraud", "ad", "normal"]

# 3. 修复缺失/损坏
python fix_missing.py    # CATEGORY = "all"

# 4. 音频增强
python post_process.py   # CATEGORY = "all"

# 5. 数据集划分
python split_data.py
```

## 编码注意

- utterance CSV → **GBK**，pandas 读时加 `encoding="gbk"`
- pattern JSON → UTF-8
- knowledge_graph CSV → UTF-8 BOM
