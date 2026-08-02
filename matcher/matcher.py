# -*- coding: utf-8 -*-
"""
净来电 · 话术模板匹配算法（B 层口径 · 生产打分）
===============================================
基于 indicates 全表 + 类型确认片段两层融合做三档权重打分。

⚠️ 口径说明（重要）：
  - 本脚本第一层使用【全量 indicates 表】（905 词，过滤 normal 命中词），
    对应 isolated_eval.py 的 B 层（图谱全量·含泄漏）
  - 因为 indicates 表中的 Brand/Phrase 实体从全量 utterance 数据提取，
    与评测数据同源，存在特征构建阶段的泄漏。
  - 所以本脚本输出的召回率（fraud 86.0% / ad 81.9%）是【生产环境最佳性能】，
    【不是】真实泛化能力。
  - 真实泛化（严格隔离）见：
      · G 层关键词（真正从 train 推导）：isolated_eval.py（fraud 72.2% / ad 64.7% / 误报 0）
      · H 层混合分类器：hybrid_clf.py（fraud 75.6% / ad 76.5% / 误报 5/120）

用途：生产部署 / 日常检查（要尽可能全的词表拦诈骗）。

权重规则：
  strong: 命中 +3 分，单次命中即可触发报警
  medium: 命中 +1 分，同一类型需 >=2 个 medium 才计入
  weak:   不进 indicates 表，留在 pattern JSON 做平滑（本算法不处理）

判定规则：
  - 对每个类型（诈骗/广告）累计得分
  - 最高分类型得分 >= THRESHOLD 时，判定为该类型
  - 同时命中诈骗和广告时取高分者

关键词来源（两层融合）：
  1. indicates 全表（从 knowledge_graph/relations.csv 加载）：过滤 normal 命中词
  2. 类型确认片段：从长词自动提取的 2-3gram，需 train 同类确认

审核承诺：
  零正常误报：所有注入词需在 800 条正常数据中零命中
  类型确认片段：仅在 train 文本上推导，不窥探 val/test

与 G 层 / H 层的关系：
  - G 层（isolated_eval.py）：关键词匹配的严格隔离版（train 推导），零误报兜底
  - H 层（hybrid_clf.py）：TF-IDF+LR 分类器 + 关键词救回，召回更高
  - 本文件（matcher.py）：生产口径（B 层全量词表），含特征泄漏，仅作日常检查

用法：
  from matcher.matcher import match_text, evaluate
  result = match_text("您的银行卡涉嫌洗钱，请转到安全账户")
  # -> ("冒充公检法人员诈骗", "fraud", 6, {"洗钱": "strong", "安全账户": "strong"})
"""
import csv
import json
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_DIR = os.path.join(BASE_DIR, "text_data")
KG_DIR = os.path.join(BASE_DIR, "text_data", "knowledge_graph")

# 打分参数
STRONG_SCORE = 3
MEDIUM_SCORE = 1
MEDIUM_MIN = 2
THRESHOLD = 3


# =========== 数据加载（延迟初始化，避免启动开销） ===========

def _load_indicates():
    """从 relations.csv 构建 word -> {type_name: weight} 表"""
    table = defaultdict(dict)
    path = os.path.join(KG_DIR, "relations.csv")
    if not os.path.exists(path):
        return table
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["relation"] != "indicates":
                continue
            word = row["head_name"]
            type_name = row["tail_name"]
            note = row.get("note", "")
            weight = "medium"
            if "weight=" in note:
                weight = note.split("weight=")[1].split(";")[0]
            table[word][type_name] = weight
    return table


def _load_type_labels():
    """从 entities.csv 判断类型是诈骗还是广告"""
    labels = {}
    path = os.path.join(KG_DIR, "entities.csv")
    if not os.path.exists(path):
        return labels
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["entity_type"] == "ScamType":
                labels[row["name"]] = "fraud"
            elif row["entity_type"] == "AdType":
                labels[row["name"]] = "ad"
    return labels


def _load_normal_texts():
    """加载全部 800 条正常话术（用于 zero-hit 验证）"""
    texts = []
    for split in ['train', 'val', 'test']:
        path = os.path.join(TEXT_DIR, "splits", "normal", f"normal_utterances_{split}.csv")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        texts.append(row["text"])
            except Exception:
                pass
    return texts


def _load_train_texts():
    """加载 fraud+ad train 数据用于类型确认片段"""
    all_train = []
    all_types = []
    for label in ['fraud', 'ad']:
        path = os.path.join(TEXT_DIR, "splits", label, f"{label}_utterances_train.csv")
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    all_train.append(row["text"])
                    all_types.append(row["type"])
    return all_train, all_types


def ngrams(s, n):
    """提取所有 n-gram"""
    return [s[i:i+n] for i in range(len(s) - n + 1)] if len(s) >= n else []


def _build_type_confirmed_fragments(indicates, min_confirm=2):
    """
    从 indicates 表中的长词（>=4字）提取类型确认短片段。

    短片段需满足：
      1. 在 train 同类文本中命中 >= min_confirm 次
      2. 在所有正常数据中零命中

    返回: {frag: {type_name: weight}}
    """
    all_train, all_types = _load_train_texts()
    if not all_train:
        return {}

    normal_texts = _load_normal_texts()
    train_types_set = set(all_types)

    # 按类型组织 train 文本
    type_to_texts = defaultdict(list)
    for text, t in zip(all_train, all_types):
        type_to_texts[t].append(text)

    # 在 train 中出现的完整 indicates 词（已覆盖的不需要片段化）
    train_vocab = set()
    for text in all_train:
        for w in indicates:
            if w in text:
                train_vocab.add(w)

    # 找出长词缺失（>=4字，不在 train 中，但零 normal 命中）
    long_missing = set()
    for w in indicates:
        if len(w) < 4:
            continue
        if w in train_vocab:
            continue
        if sum(1 for t in normal_texts if w in t) > 0:
            continue
        long_missing.add(w)

    # 提取并验证片段
    confirmed = defaultdict(dict)  # frag -> {type: weight}
    for word in long_missing:
        wtypes = indicates[word]
        for n in [2, 3]:
            for frag in set(ngrams(word, n)):
                if frag in train_vocab:
                    continue
                if sum(1 for t in normal_texts if frag in t) > 0:
                    continue
                for tn, weight in wtypes.items():
                    if tn not in train_types_set or tn not in type_to_texts:
                        continue
                    cnt = sum(1 for t in type_to_texts[tn] if frag in t)
                    if cnt >= min_confirm:
                        if tn not in confirmed[frag] or (weight == "strong" and confirmed[frag][tn] == "medium"):
                            confirmed[frag][tn] = weight

    return dict(confirmed)


# =========== 关键词表构建（两层融合） ===========

def _build_keywords():
    """
    构建两层融合关键词表。

    返回: (word_to_types, type_labels)
      word_to_types: {word: {type_name: weight}}
    """
    indicates = _load_indicates()
    type_labels = _load_type_labels()
    normal_texts = _load_normal_texts()

    # 第一层：indicates 表词（过滤掉有 normal 命中的）
    word_to_types = defaultdict(lambda: defaultdict(str))
    for word, type_map in indicates.items():
        if sum(1 for t in normal_texts if word in t) > 0:
            continue
        for tn, wt in type_map.items():
            word_to_types[word][tn] = wt

    # 第二层：类型确认片段
    fragments = _build_type_confirmed_fragments(indicates, min_confirm=2)
    for frag, type_map in fragments.items():
        for tn, wt in type_map.items():
            word_to_types[frag][tn] = wt

    return dict(word_to_types), type_labels


# =========== 公开 API ===========


def match_text(text):
    """
    对一段文本做话术匹配（两层融合关键词表：indicates 全表 + 类型确认片段）。

    参数:
        text: str, 待匹配的文本

    返回:
        (best_type, label, best_score, hits)
        或 (None, None, 0, {})
    """
    if not text:
        return None, None, 0, {}

    # 延迟加载
    global _WORD_TO_TYPES, _TYPE_LABELS
    if "_WORD_TO_TYPES" not in globals():
        _WORD_TO_TYPES, _TYPE_LABELS = _build_keywords()

    type_scores = defaultdict(int)
    type_medium = defaultdict(int)
    hits = {}

    for word, type_weights in _WORD_TO_TYPES.items():
        if word in text:
            for type_name, weight in type_weights.items():
                if weight == "strong":
                    type_scores[type_name] += STRONG_SCORE
                elif weight == "medium":
                    type_medium[type_name] += 1
                hits[word] = (type_name, weight)

    for type_name, cnt in type_medium.items():
        if cnt >= MEDIUM_MIN:
            type_scores[type_name] += MEDIUM_SCORE * cnt

    if not type_scores:
        return None, None, 0, hits

    best_type = max(type_scores, key=type_scores.get)
    best_score = type_scores[best_type]
    if best_score < THRESHOLD:
        return None, None, best_score, hits

    label = _TYPE_LABELS.get(best_type, "unknown")
    return best_type, label, best_score, hits


def reload():
    """强制重新加载关键词表（数据更新后调用）"""
    global _WORD_TO_TYPES, _TYPE_LABELS
    _WORD_TO_TYPES, _TYPE_LABELS = _build_keywords()
    return len(_WORD_TO_TYPES)


# =========== 验收评测 ===========

def _load_utterances(label):
    """读取某一类的全部话术文本和子类型"""
    path = os.path.join(TEXT_DIR, f"{label}_utterances.csv")
    texts, types_ = [], []
    try:
        with open(path, encoding="gbk") as f:
            for row in csv.DictReader(f):
                texts.append(row["text"])
                types_.append(row["type"])
    except Exception:
        try:
            with open(path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    texts.append(row["text"])
                    types_.append(row["type"])
        except Exception:
            return [], []
    return texts, types_


def evaluate():
    """生产口径验收（B 层全量词表，含特征泄漏，仅日常检查）。"""
    print("话术模板匹配算法验收 B层口径")
    results = {}
    for label, expect_label in [("fraud", "fraud"), ("ad", "ad"), ("normal", "normal")]:
        texts, types_ = _load_utterances(label)
        tp = 0
        total = len(texts)
        det_label = defaultdict(int)
        wrong_label = []
        detected_types = defaultdict(int)
        type_total = defaultdict(int)

        for i, (text, t) in enumerate(zip(texts, types_)):
            type_total[t] += 1
            best_type, pred_label, score, hits = match_text(text)
            if pred_label == expect_label:
                tp += 1
                if best_type:
                    detected_types[t] += 1
                if best_type == t:
                    det_label[t] += 1
            else:
                wrong_label.append((i, text[:30], pred_label, best_type, score))

        recall = tp / total * 100 if total > 0 else 0
        results[label] = {
            "total": total, "hit": tp, "recall": recall,
            "type_total": dict(type_total),
            "type_hit": dict(detected_types),
            "type_correct": dict(det_label),
            "wrong": wrong_label,
        }

        print(f"\n[{label}] 样本 {total} 条，命中 {tp} 条，召回率 {recall:.1f}%")
        if label == "normal":
            # normal 的正确行为是「未命中任何诈骗/广告类型」= 拒识
            # 误报 = 被判成 fraud/ad 的 normal 样本
            fps = [(i, txt, pl, bt, sc) for i, txt, pl, bt, sc in wrong_label if pl is not None]
            print(f"    误报: {len(fps)} 条（normal 被判成 fraud/ad）")
            for i, txt, pl, bt, sc in fps[:8]:
                print(f"      #{i} '{txt}' -> {pl}({bt}) score={sc}")
        else:
            zero_types = [t for t, tot in type_total.items() if detected_types.get(t, 0) == 0]
            print(f"    类型覆盖: {len([t for t in type_total if detected_types.get(t,0)>0])}/{len(type_total)}")
            if zero_types:
                print(f"    0 命中类型: {zero_types}")
            total_type_hit = sum(detected_types.values())
            total_type_correct = sum(det_label.values())
            if total_type_hit > 0:
                acc = total_type_correct / total_type_hit * 100
                print(f"    类型准确率: {acc:.1f}%")
                confusion = defaultdict(int)
                for i, txt, pl, bt, sc in wrong_label:
                    if pl == expect_label and bt is not None and bt != types_[i]:
                        confusion[f"{types_[i]}->{bt}"] += 1
                top_conf = sorted(confusion.items(), key=lambda x: -x[1])[:5]
                if top_conf:
                    print(f"    主要混淆: {top_conf}")

    print("\n" + "=" * 64)
    # 正常误报 = normal 被判成 fraud/ad 的样本数
    normal_fp = len([w for w in results['normal']['wrong'] if w[2] is not None])
    print(f"总结: 诈骗召回 {results['fraud']['recall']:.1f}% | "
          f"广告召回 {results['ad']['recall']:.1f}% | "
          f"正常误报 {normal_fp}/{results['normal']['total']} 条")
    print("=" * 64)

    return results


if __name__ == "__main__":
    n = reload()
    print(f"已加载 {n} 个关键词（indicates 全表 + 类型确认片段）")
    print()
    evaluate()
