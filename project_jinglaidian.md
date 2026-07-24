---
name: jinglaidian-scam-call-project
description: 净来电 — 诈骗电话识别校赛项目，成员A负责数据线（TTS、话术、标注、模式库），截至2026.7.30共1091条数据
type: project
---

净来电项目 — 全国人工智能算法挑战赛校赛。

**成员A角色：** 数据线，负责 TTS 合成语音、话术编写、标注质量检查、模式库构建。

**数据量（截至7.30）：** 诈骗371条（28种类型）、广告262条（14种类型）、正常458条（正常通话454+同事沟通4），合计1091条。零文本重叠、零标注错误。

**代码目录：** `C:\Users\Subway\Desktop\诈骗电话识别\TTS\`，含 config.py / main.py / post_process.py / fix_missing.py / fix_one.py / split_data.py 六个脚本，不新增脚本文件。

**数据文件：** `C:\Users\Subway\Desktop\诈骗电话识别\text_data\`
- fraud_utterances.csv (label=fraud, GBK编码)
- ad_utterances.csv (label=ad, GBK编码)
- normal_utterances.csv (label=normal, GBK编码)

**模式库：** `text_data/pattern_library/`
- fraud_patterns.json: 28种模板，4组（权威压迫/情感诱导/利益诱惑/服务伪装）
- ad_patterns.json: 14种模板，2组（回访跟进/推介直给），每种≥2个模板模式
- normal_patterns.json: 2种类型，8个模板，18条confusable_boundaries

**关键约束：**
- 8.1 数据冻结，之后不能再改 CSV
- CSV 是 GBK 编码，不要改编码
- 按 label 字段判断类别，不要按文件名猜
- pattern JSON 关键词必须完整（不能有3-4字碎片如"您的快"）
- 所有功能在现有6个 .py 文件里完成，不新增脚本
- 用户是校赛选手，解释要直白
- **广告和正常的灰色地带要仔细甄别：回访型售后服务/课程回访 ≠ 广告推销，拿不准要问用户**

**8.1之后计划：** 话术知识图谱（8.2-8.3）→ 模板匹配算法（8.4-8.5）→ 难例挖掘（8.6-8.7）→ 知识图谱-模型联动（8.8-8.9）→ 根据模型反馈迭代数据（8.10）
