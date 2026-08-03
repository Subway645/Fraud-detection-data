# -*- coding: utf-8 -*-
"""
净来电 · 话术模板匹配算法（B 层口径 · 生产打分）
================================================
基于 indicates 全表的三档权重打分器。

口径说明
--------
本脚本即 isolated_eval.py 的 B 层（图谱全量·含泄漏）：
  - 使用【全量 indicates 表】（905 词），不做 normal 过滤，不做类型确认片段
  - 因 indicates 表中的 Brand/Phrase 实体从全量 utterance 提取，与评测数据
    同源，存在特征构建阶段的泄漏
  - 输出（fraud 88.3% / ad 80.8% / normal 误报 22/800）为生产环境最佳性能，
    【不是】真实泛化能力
真实泛化（严格隔离）见 isolated_eval.py：
  · G 层关键词（train 推导 + val 调参）：fraud 62.2% / ad 64.7% / 误报 3/120
  · H 层混合分类器：hybrid_clf.py：fraud 80.0% / ad 85.3% / 误报 5/120

权重规则
--------
  strong  命中 +3 分，单次命中即可触发报警
  medium  命中 +1 分，同一类型需 >= 2 个 medium 才计入
  weak    不进 indicates 表，仅用于平分类别时的语义取向（见 match_text）

判定规则
--------
  1. 对每个类型累计得分，最高分 >= THRESHOLD 时判定为该类型
  2. 诈骗/广告同时命中时取高分者
  3. 平分类别时按确定性规则决胜（weak 命中数 → 广告优先 → 类型名排序）

用法
----
  from matcher.matcher import match_text
  match_text("您的银行卡涉嫌洗钱，请转到安全账户")
  # -> ("冒充公检法人员诈骗", "fraud", 6, {"洗钱": "strong", "安全账户": "strong"})
"""
import csv
import json
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_DIR = os.path.join(BASE_DIR, "text_data")
KG_DIR = os.path.join(BASE_DIR, "text_data", "knowledge_graph")

STRONG_SCORE = 3
MEDIUM_SCORE = 1
MEDIUM_MIN = 2
THRESHOLD = 3


# =========== 数据加载 ===========

def _load_indicates():
    """构建 word -> {type_name: weight} 映射（来自 relations.csv 的 indicates 关系）。"""
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
    """构建 type_name -> (fraud|ad) 映射（来自 entities.csv 的类型实体）。"""
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


def _load_weak():
    """加载 pattern JSON 中的 weak 词（type -> [words]）。

    weak 词不加分、不进 indicates 表，仅用于平分类别时的语义取向：
    同分时命中 weak 词更多的类型在题材上更贴近。
    """
    weak = defaultdict(list)
    for pfile in ["fraud_patterns.json", "ad_patterns.json"]:
        path = os.path.join(TEXT_DIR, "pattern_library", pfile)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for type_name, pat in data.get("patterns", {}).items():
            for kw in pat.get("keywords", []):
                if kw.get("weight") == "weak":
                    weak[type_name].append(kw["word"])
    return dict(weak)


def _weak_hits(text, type_name, weak_table):
    """统计 text 命中某类型 weak 词的个数。"""
    return sum(1 for w in weak_table.get(type_name, []) if w in text)


# =========== 关键词表构建（B 层：全量 indicates，不过滤） ===========

def _build_keywords():
    """构建关键词表：indicates 全表原样使用，不做 normal 过滤，不做片段。"""
    indicates = _load_indicates()
    type_labels = _load_type_labels()

    word_to_types = defaultdict(lambda: defaultdict(str))
    for word, type_map in indicates.items():
        for tn, wt in type_map.items():
            word_to_types[word][tn] = wt

    return dict(word_to_types), type_labels


# =========== 公开 API ===========

def match_text(text):
    """对文本做话术匹配（indicates 全表）。

    返回 (best_type, label, best_score, hits)；未触发报警时返回 (None, None, 0, hits)。
    """
    if not text:
        return None, None, 0, {}

    global _WORD_TO_TYPES, _TYPE_LABELS, _WEAK
    if "_WORD_TO_TYPES" not in globals():
        _WORD_TO_TYPES, _TYPE_LABELS = _build_keywords()
        _WEAK = _load_weak()

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

    # 平分类别时按确定性规则决胜（与 isolated_eval.py / hybrid_clf.py 统一）
    max_score = max(type_scores.values())
    best_candidates = [tn for tn, sc in type_scores.items() if sc == max_score]
    if len(best_candidates) > 1:
        wc = {tn: _weak_hits(text, tn, _WEAK) for tn in best_candidates}
        max_weak = max(wc.values())
        top = [tn for tn in best_candidates if wc[tn] == max_weak] if max_weak > 0 else best_candidates
        ad_cands = [tn for tn in top if _TYPE_LABELS.get(tn) == "ad"]
        best_type = ad_cands[0] if ad_cands else sorted(top)[0]
    else:
        best_type = best_candidates[0]

    best_score = type_scores[best_type]
    if best_score < THRESHOLD:
        return None, None, best_score, hits

    label = _TYPE_LABELS.get(best_type, "unknown")
    return best_type, label, best_score, hits


def reload():
    """重新构建关键词表（数据更新后调用），返回关键词总数。"""
    global _WORD_TO_TYPES, _TYPE_LABELS, _WEAK
    _WORD_TO_TYPES, _TYPE_LABELS = _build_keywords()
    _WEAK = _load_weak()
    return len(_WORD_TO_TYPES)


# =========== B 层口径评测 ===========

def _load_utterances(label):
    """读取某一类别的全部话术文本及子类型。"""
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
    """B 层口径验收（indicates 全量，含特征泄漏，仅日常检查）。"""
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
            # normal 的正确行为是未命中任何诈骗/广告类型（拒识）；误报即被判成 fraud/ad
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
    normal_fp = len([w for w in results['normal']['wrong'] if w[2] is not None])
    print(f"总结: 诈骗召回 {results['fraud']['recall']:.1f}% | "
          f"广告召回 {results['ad']['recall']:.1f}% | "
          f"正常误报 {normal_fp}/{results['normal']['total']} 条")
    print("=" * 64)

    return results


if __name__ == "__main__":
    n = reload()
    print(f"已加载 {n} 个关键词（indicates 全表）")
    print()
    evaluate()
