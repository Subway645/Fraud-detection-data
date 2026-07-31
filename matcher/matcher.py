# -*- coding: utf-8 -*-
"""
净来电 · 话术模板匹配算法
==========================
基于 knowledge_graph/relations.csv 的 indicates 关系做三档权重打分。

权重规则：
  strong: 命中 +3 分，单次命中即可触发报警
  medium: 命中 +1 分，同一类型需 >=2 个 medium 才计入
  weak:   不进 indicates 表，留在 pattern JSON 做平滑（本算法不处理）

判定规则：
  - 对每个类型（诈骗/广告）累计得分
  - 最高分类型得分 >= THRESHOLD 时，判定为该类型
  - 同时命中诈骗和广告时取高分者

用法：
  from matcher import match_text, evaluate
  result = match_text("您的银行卡涉嫌洗钱，请转到安全账户")
  # → ("冒充公检法人员诈骗", "fraud", 6, {"洗钱": "strong", "安全账户": "strong"})
"""
import csv
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_DIR = os.path.join(BASE_DIR, "text_data", "knowledge_graph")

# 打分参数
STRONG_SCORE = 3
MEDIUM_SCORE = 1
MEDIUM_MIN = 2          # medium 至少几个才算数
THRESHOLD = 3           # 触发报警的最低总分（= 1 个 strong，或 3 个 medium）


def _load_indicates():
    """从 relations.csv 构建 word -> {type_name: weight} 表"""
    table = defaultdict(dict)
    path = os.path.join(KG_DIR, "relations.csv")
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["relation"] != "indicates":
                continue
            word = row["head_name"]
            type_name = row["tail_name"]
            note = row["note"]
            weight = "medium"
            if "weight=" in note:
                weight = note.split("weight=")[1].split(";")[0]
            table[word][type_name] = weight
    return table


def _load_type_labels():
    """从 entities.csv 判断类型是诈骗还是广告"""
    labels = {}
    path = os.path.join(KG_DIR, "entities.csv")
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["entity_type"] == "ScamType":
                labels[row["name"]] = "fraud"
            elif row["entity_type"] == "AdType":
                labels[row["name"]] = "ad"
    return labels


INDICATES = _load_indicates()
TYPE_LABELS = _load_type_labels()


def match_text(text):
    """
    对一段文本做话术匹配。

    返回 (best_type, label, best_score, hits) 或 (None, None, 0, {})
    hits: {word: (type, weight)}
    """
    if not text:
        return None, None, 0, {}

    # 每类型累计
    type_scores = defaultdict(int)      # type -> 总分
    type_medium = defaultdict(int)      # type -> medium 命中数
    type_strong = defaultdict(int)      # type -> strong 命中数
    hits = {}                           # word -> (type, weight)

    for word, type_weights in INDICATES.items():
        if word in text:
            for type_name, weight in type_weights.items():
                if weight == "strong":
                    type_scores[type_name] += STRONG_SCORE
                    type_strong[type_name] += 1
                elif weight == "medium":
                    type_medium[type_name] += 1
                hits[word] = (type_name, weight)

    # medium 需 >=2 个才算
    for type_name, cnt in type_medium.items():
        if cnt >= MEDIUM_MIN:
            type_scores[type_name] += MEDIUM_SCORE * cnt

    if not type_scores:
        return None, None, 0, hits

    # 取最高分
    best_type = max(type_scores, key=type_scores.get)
    best_score = type_scores[best_type]
    if best_score < THRESHOLD:
        return None, None, best_score, hits

    label = TYPE_LABELS.get(best_type, "unknown")
    return best_type, label, best_score, hits


def _load_utterances(label):
    """读取某一类的全部话术文本和子类型"""
    path = os.path.join(BASE_DIR, "text_data", f"{label}_utterances.csv")
    texts = []
    types = []
    with open(path, encoding="gbk") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            types.append(row["type"])
    return texts, types


def evaluate():
    """用三类 utterance CSV 验收：召回率 / 误报率 / 分类准确率 / 类型覆盖"""
    print("=" * 64)
    print("话术模板匹配算法 · 图谱健康性验收")
    print("=" * 64)

    results = {}

    for label, expect_label in [("fraud", "fraud"), ("ad", "ad"), ("normal", "normal")]:
        texts, types = _load_utterances(label)
        tp = 0          # 正确报警（fraud/ad 被报成对应类）
        total = len(texts)
        det_label = defaultdict(int)
        wrong_label = []
        detected_types = defaultdict(int)
        type_total = defaultdict(int)

        for i, (text, t) in enumerate(zip(texts, types)):
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

        recall = tp / total * 100
        results[label] = {
            "total": total,
            "hit": tp,
            "recall": recall,
            "type_total": dict(type_total),
            "type_hit": dict(detected_types),
            "type_correct": dict(det_label),
            "wrong": wrong_label,
        }

        print(f"\n[{label}] 样本 {total} 条，命中 {tp} 条，召回率 {recall:.1f}%")
        if label == "normal":
            print(f"    误报: {len(wrong_label)} 条")
            for i, txt, pl, bt, sc in wrong_label[:8]:
                print(f"      #{i} '{txt}' → {pl}({bt}) score={sc}")
        else:
            # 类型覆盖
            zero_types = [t for t, tot in type_total.items() if detected_types.get(t, 0) == 0]
            print(f"    类型覆盖: {len([t for t in type_total if detected_types.get(t,0)>0])}/{len(type_total)}")
            if zero_types:
                print(f"    0 命中类型: {zero_types}")
            # 分类准确率（在被正确判为大类的前提下）
            total_type_hit = sum(detected_types.values())
            total_type_correct = sum(det_label.values())
            if total_type_hit > 0:
                acc = total_type_correct / total_type_hit * 100
                print(f"    类型准确率(命中里判对子类型): {acc:.1f}%")
                # 混淆最多的
                confusion = defaultdict(int)
                for i, txt, pl, bt, sc in wrong_label:
                    if pl == expect_label and bt != None and bt != types[i]:
                        confusion[f"{types[i]}→{bt}"] += 1
                top_conf = sorted(confusion.items(), key=lambda x: -x[1])[:5]
                if top_conf:
                    print(f"    主要混淆: {top_conf}")

    print("\n" + "=" * 64)
    print(f"总结: 诈骗召回 {results['fraud']['recall']:.1f}% | "
          f"广告召回 {results['ad']['recall']:.1f}% | "
          f"正常误报 {results['normal']['hit']}/{results['normal']['total']} 条")
    print("=" * 64)

    return results


if __name__ == "__main__":
    evaluate()
