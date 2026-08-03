# -*- coding: utf-8 -*-
"""
净来电 · 话术模板匹配算法 · 隔离评测
====================================
对七组策略在统一协议下评测，核心目标为报告"真实泛化能力"（严格隔离）。

评测层
------
  A. 模板词基线       (全量参考 · 含泄漏)
  B. 图谱全量         (全量参考 · 含泄漏)
  C. 类型过滤         (train → test)
  D. 严格隔离         (train → test)
  E. 完美词隔离       (train → test)
  G. 类型确认片段     (train → val 调参 → test · 关键词匹配最终)
  H. 混合分类器       (train → val 调参 → test · 整体最终)

严格隔离口径
------------
特征词仅从 train 推导，超参仅由 val 确定，最终指标只报告独立 test
（fraud 90 / ad 68 / normal 120）。normal 特征筛选只用 normal 的 train，
不使用 val/test，避免特征构建泄漏。
"""
import csv, json, os
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_DIR = os.path.join(BASE_DIR, "text_data")

STRONG_SCORE = 3; MEDIUM_SCORE = 1; MEDIUM_MIN = 2; THRESHOLD = 3


# =========== 数据加载 ===========

def load_splits(label):
    """加载某类别 train/val/test 三份划分的文本与子类型。"""
    result = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(TEXT_DIR, "splits", label, f"{label}_utterances_{split}.csv")
        texts, types = [], []
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                texts.append(row["text"]); types.append(row["type"])
        result[split] = (texts, types)
    return result


def load_pattern_keywords():
    """从 pattern JSON 加载人工设计的 strong/medium 关键词（A 层基线）。"""
    keywords = defaultdict(set); weights = {}
    for pfile in ["fraud_patterns.json", "ad_patterns.json"]:
        path = os.path.join(TEXT_DIR, "pattern_library", pfile)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for tn, pat in data["patterns"].items():
            for kw in pat.get("keywords", []):
                if kw["weight"] in ("strong", "medium"):
                    keywords[tn].add(kw["word"])
                    w = kw["word"]
                    if w not in weights: weights[w] = kw["weight"]
                    elif kw["weight"] == "strong" and weights[w] == "medium":
                        weights[w] = "strong"
    return keywords, weights


def load_indicates():
    """加载 indicates 关系：word -> {type_name: weight}。"""
    word_types = defaultdict(dict)
    path = os.path.join(TEXT_DIR, "knowledge_graph", "relations.csv")
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["relation"] != "indicates": continue
            w = row["head_name"]
            weight = "strong" if "weight=strong" in row.get("note", "") else "medium"
            word_types[w][row["tail_name"]] = weight
    return word_types


def load_type_labels():
    """加载类型实体到类别（fraud/ad）的映射。"""
    labels = {}
    path = os.path.join(TEXT_DIR, "knowledge_graph", "entities.csv")
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["entity_type"] == "ScamType": labels[row["name"]] = "fraud"
            elif row["entity_type"] == "AdType": labels[row["name"]] = "ad"
    return labels


def load_normal_train():
    """加载 normal 的 train 话术，用于特征词筛选（零命中验证）。

    严格隔离：只用 train，不使用 val/test，避免偷看 normal 测试集。
    """
    texts = []
    path = os.path.join(TEXT_DIR, "splits", "normal", "normal_utterances_train.csv")
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
    return texts


# =========== 匹配引擎 ===========

_WEAK = None  # 懒加载 weak 词表 {type: [words]}

def _get_weak():
    global _WEAK
    if _WEAK is None:
        _WEAK = {}
        for pfile in ["fraud_patterns.json", "ad_patterns.json"]:
            path = os.path.join(TEXT_DIR, "pattern_library", pfile)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for type_name, pat in data.get("patterns", {}).items():
                for kw in pat.get("keywords", []):
                    if kw.get("weight") == "weak":
                        _WEAK.setdefault(type_name, []).append(kw["word"])
    return _WEAK


def match_text(text, keywords, weights, type_labels, threshold=THRESHOLD):
    """关键词打分并判定类型。

    返回 (best_type, label, score)；未触发报警时返回 (None, None, 0)。
    平分类别时按确定性规则决胜（weak 命中数 → 广告优先 → 类型名排序），
    与 matcher.py / hybrid_clf.py 统一，消除跨进程哈希依赖。
    """
    if not text:
        return None, None, 0
    type_scores = defaultdict(int); type_med = defaultdict(int)
    for tn, words in keywords.items():
        for w in words:
            if w in text:
                wt = weights.get(w, "medium")
                if wt == "strong":
                    type_scores[tn] += STRONG_SCORE
                else:
                    type_med[tn] += 1
    for tn, c in type_med.items():
        if c >= MEDIUM_MIN:
            type_scores[tn] += MEDIUM_SCORE * c
    if not type_scores:
        return None, None, 0

    max_score = max(type_scores.values())
    best_candidates = [tn for tn, sc in type_scores.items() if sc == max_score]
    if len(best_candidates) > 1:
        weak = _get_weak()
        wc = {tn: sum(1 for w in weak.get(tn, []) if w in text) for tn in best_candidates}
        max_weak = max(wc.values())
        top = [tn for tn in best_candidates if wc[tn] == max_weak] if max_weak > 0 else best_candidates
        ad_cands = [tn for tn in top if type_labels.get(tn) == "ad"]
        best = ad_cands[0] if ad_cands else sorted(top)[0]
    else:
        best = best_candidates[0]
    if type_scores[best] < threshold:
        return None, None, type_scores[best]
    return best, type_labels.get(best, "unknown"), type_scores[best]


# =========== 评测函数 ===========

def evaluate_layer(texts, types, expect_label, keywords, weights, type_labels, verbose=False,
                   threshold=THRESHOLD):
    """在给定文本集上评测召回率、类型覆盖、类型准确率与误报数。"""
    total = len(texts)
    tp = 0; dtypes = defaultdict(int); ctypes = defaultdict(int)
    ttotal = defaultdict(int); wrong = []; fps = []
    for i, (text, t) in enumerate(zip(texts, types)):
        ttotal[t] += 1
        best_type, pred_label, score = match_text(text, keywords, weights, type_labels, threshold)
        if pred_label == expect_label:
            tp += 1
            if best_type: dtypes[best_type] += 1
            if best_type == t: ctypes[t] += 1
        else:
            if pred_label is not None:
                fps.append((i, text[:60], pred_label, best_type, score))
            wrong.append((i, text[:30], pred_label, best_type, score, t))
    recall = tp / total * 100 if total > 0 else 0
    zero_types = [t for t, _ in ttotal.items() if dtypes.get(t, 0) == 0]
    tc = f"{len(ttotal) - len(zero_types)}/{len(ttotal)}"
    total_d = sum(dtypes.values()); total_c = sum(ctypes.values())
    acc = total_c / total_d * 100 if total_d > 0 else 0
    confusion = Counter()
    for i, txt, pl, bt, sc, true_t in wrong:
        if pl == expect_label and bt is not None and bt != true_t:
            confusion[f"{true_t}->{bt}"] += 1
    if verbose and fps:
        for idx, txt, pl, bt, sc in fps[:10]:
            print(f"      #{idx} '{txt}' -> {pl}({bt}) score={sc}")
    return {"total": total, "hit": tp, "recall": recall,
            "type_coverage": tc, "zero_hit_types": zero_types,
            "type_accuracy": acc, "confusions": confusion.most_common(5),
            "fp_count": len(fps), "false_positives": fps[:10]}


# =========== 关键词构建策略 ===========

def ngrams(s, n):
    """提取文本 s 的所有长度为 n 的子串。"""
    return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else []


def build_strict_kw(train_texts, train_types, indicates):
    """D 层：仅保留 train 中实际出现的 indicates 词。"""
    ts = set(train_types); vocab = set()
    for text in train_texts:
        for w in indicates:
            if w in text: vocab.add(w)
    kw = defaultdict(set); ww = {}
    for w in vocab:
        for tn, wt in indicates[w].items():
            if tn in ts:
                kw[tn].add(w)
                if w not in ww: ww[w] = wt
                elif wt == "strong" and ww[w] == "medium": ww[w] = "strong"
    return kw, ww


def build_type_filter_kw(train_types, indicates):
    """C 层：按 train 出现的类型过滤 indicates 全表（词仍可能含 test 信息）。"""
    ts = set(train_types); kw = defaultdict(set); ww = {}
    for w, type_map in indicates.items():
        for tn, wt in type_map.items():
            if tn in ts:
                kw[tn].add(w)
                if w not in ww: ww[w] = wt
                elif wt == "strong" and ww[w] == "medium": ww[w] = "strong"
    return kw, ww


def build_perfect_kw(all_train_texts, all_train_types, indicates, normal_texts):
    """E 层：train 出现 + 对 normal 零命中的 indicates 词。"""
    ts = set(all_train_types)
    nh = {w: sum(1 for t in normal_texts if w in t) for w in indicates}
    vocab = set()
    for text in all_train_texts:
        for w in indicates:
            if w in text: vocab.add(w)
    perfect = {w for w in vocab if nh.get(w, 0) == 0}
    kw = defaultdict(set); ww = {}
    for w in perfect:
        for tn, wt in indicates[w].items():
            if tn in ts:
                kw[tn].add(w)
                if w not in ww: ww[w] = wt
                elif wt == "strong" and ww[w] == "medium": ww[w] = "strong"
    return kw, ww, {"perfect_total": len(perfect),
                     "filtered_by_normal": len(vocab) - len(perfect)}


def build_perfect_with_fragments(all_train_texts, all_train_types, indicates, normal_texts, min_confirm=2):
    """G 层：完美词 + 类型确认片段。

    类型确认片段从 indicates 长词（>=4 字）中枚举 2/3-gram 子串，
    要求在该类型 train 文本中出现 >= min_confirm 次、且对 normal 零命中。
    """
    ts = set(all_train_types)
    t2t = defaultdict(list)
    for text, t in zip(all_train_texts, all_train_types):
        t2t[t].append(text)
    nh = {w: sum(1 for t in normal_texts if w in t) for w in indicates}
    vocab = set()
    for text in all_train_texts:
        for w in indicates:
            if w in text: vocab.add(w)
    perfect = {w for w in vocab if nh.get(w, 0) == 0}
    kw = defaultdict(set); ww = {}
    for w in perfect:
        for tn, wt in indicates[w].items():
            if tn in ts:
                kw[tn].add(w)
                if w not in ww: ww[w] = wt
                elif wt == "strong" and ww[w] == "medium": ww[w] = "strong"

    long_missing = {w for w in indicates if len(w) >= 4 and w not in vocab and nh.get(w, 0) == 0}
    conf_frags = defaultdict(set)
    for word in long_missing:
        wtypes = set(indicates[word].keys())
        for n in [2, 3]:
            for frag in set(ngrams(word, n)):
                if frag in ww: continue
                if sum(1 for t in normal_texts if frag in t) > 0: continue
                for tn in wtypes:
                    if tn in ts and tn in t2t:
                        if sum(1 for t in t2t[tn] if frag in t) >= min_confirm:
                            conf_frags[frag].add(tn)
    frag_added = 0
    for frag, tns in conf_frags.items():
        for tn in tns:
            if tn in ts:
                kw[tn].add(frag)
                best = "medium"
                for word in long_missing:
                    if frag in word and tn in indicates[word]:
                        if indicates[word][tn] == "strong": best = "strong"
                ww[frag] = best; frag_added += 1
    return kw, ww, {"perfect_total": len(perfect),
                     "long_missing": len(long_missing),
                     "fragments_confirmed": len(conf_frags),
                     "fragments_added": frag_added,
                     "total_words": len(ww)}


# =========== 主评测 ===========

def bar(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def main():
    type_labels = load_type_labels()
    indicates = load_indicates()
    pattern_kw, pattern_ww = load_pattern_keywords()
    normal_train = load_normal_train()

    fraud = load_splits("fraud"); ad = load_splits("ad"); norm = load_splits("normal")

    # 全量语料（A/B 层参考用）
    fall_t = fraud["train"][0] + fraud["val"][0] + fraud["test"][0]
    fall_ty = fraud["train"][1] + fraud["val"][1] + fraud["test"][1]
    aall_t = ad["train"][0] + ad["val"][0] + ad["test"][0]
    aall_ty = ad["train"][1] + ad["val"][1] + ad["test"][1]
    nall_t = norm["train"][0] + norm["val"][0] + norm["test"][0]
    nall_ty = norm["train"][1] + norm["val"][1] + norm["test"][1]

    all_kw = defaultdict(set); all_ww = {}
    for w, tm in indicates.items():
        for tn, wt in tm.items():
            all_kw[tn].add(w)
            if w not in all_ww: all_ww[w] = wt
            elif wt == "strong" and all_ww[w] == "medium": all_ww[w] = "strong"

    bar("话术模板匹配 · 隔离评测（A-E+G 关键词线 + H 混合分类）")
    print(f"splits 7:1.5:1.5 seed=42 | strong={STRONG_SCORE} medium={MEDIUM_SCORE} thresh={THRESHOLD}")
    print(f"Indicates: {len(indicates)}词  Pattern: {len(pattern_ww)}词")

    # ---- A: 模板词基线 ----
    bar("A. 模板词基线")
    for label, texts, types_ in [("fraud", fall_t, fall_ty), ("ad", aall_t, aall_ty)]:
        r = evaluate_layer(texts, types_, label, pattern_kw, pattern_ww, type_labels)
        print(f"  [{label}] 召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})  "
              f"覆盖率 {r['type_coverage']}  准确率 {r['type_accuracy']:.1f}%")
    r = evaluate_layer(nall_t, nall_ty, "normal", pattern_kw, pattern_ww, type_labels)
    print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")

    # ---- B: 图谱全量（含泄漏） ----
    bar("B. 图谱全量 (含泄漏)")
    for label, texts, types_ in [("fraud", fall_t, fall_ty), ("ad", aall_t, aall_ty)]:
        r = evaluate_layer(texts, types_, label, all_kw, all_ww, type_labels)
        print(f"  [{label}] 召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})  "
              f"覆盖率 {r['type_coverage']}  准确率 {r['type_accuracy']:.1f}%")
        if r["zero_hit_types"]:
            print(f"    零命中: {r['zero_hit_types']}")
    r = evaluate_layer(nall_t, nall_ty, "normal", all_kw, all_ww, type_labels)
    print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")

    # ---- C: 类型过滤（残余泄漏） ----
    bar("C. 类型过滤 (残余泄漏)")
    for label, s in [("fraud", fraud), ("ad", ad)]:
        tk, tw = build_type_filter_kw(s["train"][1], indicates)
        te_t = s["test"][0]; te_ty = s["test"][1]
        r = evaluate_layer(te_t, te_ty, label, tk, tw, type_labels)
        print(f"  [{label}] 词数{len(tw)}  召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})  "
              f"覆盖率 {r['type_coverage']}  准确率 {r['type_accuracy']:.1f}%")
    tk_f, tw_f = build_type_filter_kw(fraud["train"][1], indicates)
    tk_a, tw_a = build_type_filter_kw(ad["train"][1], indicates)
    c_kw = defaultdict(set); c_ww = {}
    for d in [tk_f, tk_a]:
        for tn, ws in d.items():
            c_kw[tn] |= ws
    c_ww.update(tw_f); c_ww.update(tw_a)
    nte_t = norm["test"][0]; nte_ty = norm["test"][1]
    r = evaluate_layer(nte_t, nte_ty, "normal", c_kw, c_ww, type_labels)
    print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")

    # ---- D: 严格隔离（真实下限） ----
    bar("D. 严格隔离 (真实下限)")
    for label, s in [("fraud", fraud), ("ad", ad)]:
        sk, sw = build_strict_kw(s["train"][0], s["train"][1], indicates)
        te_t = s["test"][0]; te_ty = s["test"][1]
        r = evaluate_layer(te_t, te_ty, label, sk, sw, type_labels)
        print(f"  [{label}] 词数{len(sw)}  召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})  "
              f"覆盖率 {r['type_coverage']}  准确率 {r['type_accuracy']:.1f}%")
        if r["zero_hit_types"]:
            print(f"    零命中: {r['zero_hit_types']}")
    sk_f, sw_f = build_strict_kw(fraud["train"][0], fraud["train"][1], indicates)
    sk_a, sw_a = build_strict_kw(ad["train"][0], ad["train"][1], indicates)
    s_kw = defaultdict(set); s_ww = {}
    for d in [sk_f, sk_a]:
        for tn, ws in d.items():
            s_kw[tn] |= ws
    s_ww.update(sw_f); s_ww.update(sw_a)
    r = evaluate_layer(nte_t, nte_ty, "normal", s_kw, s_ww, type_labels)
    print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")

    # ---- E: 完美词（零误报基准） ----
    bar("E. 完美词隔离 (零误报基准)")
    perf = {}
    for label, s in [("fraud", fraud), ("ad", ad)]:
        pk, pw, ps = build_perfect_kw(s["train"][0], s["train"][1], indicates, normal_train)
        perf[label] = (pk, pw, ps)
        te_t = s["test"][0]; te_ty = s["test"][1]
        tr = evaluate_layer(s["train"][0], s["train"][1], label, pk, pw, type_labels)
        r = evaluate_layer(te_t, te_ty, label, pk, pw, type_labels)
        print(f"  [{label}] 完美词{ps['perfect_total']} (过滤{ps['filtered_by_normal']})")
        print(f"          train: 召回 {tr['recall']:.1f}%  test: 召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})  "
              f"覆盖率 {r['type_coverage']}  准确率 {r['type_accuracy']:.1f}%")
        if r["zero_hit_types"]:
            print(f"          零命中: {r['zero_hit_types']}")
    p_kw = defaultdict(set); p_ww = {}
    for label_ in ["fraud", "ad"]:
        pk, pw, _ = perf[label_]
        for tn, ws in pk.items():
            p_kw[tn] |= ws
        p_ww.update(pw)
    r = evaluate_layer(nte_t, nte_ty, "normal", p_kw, p_ww, type_labels)
    print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")

    # ---- G: 类型确认片段（val 调参 · 关键词最终） ----
    bar("G. 类型确认片段 (val 调参 · 关键词最终)")
    # 超参：min_confirm（片段需同类命中 >= 几次）、threshold（触发报警分数）
    # 规范做法：val 上网格扫描选最优，再用最优参数在 test 上评测
    best_g_cfg = None  # (score, min_confirm, threshold)
    for mc in [1, 2, 3]:
        for th in [3, 4, 5, 6]:
            val_f_hits = 0; val_f_total = 0
            val_a_hits = 0; val_a_total = 0
            val_n_fp = 0; val_n_total = 0
            for label, s in [("fraud", fraud), ("ad", ad)]:
                gk_tmp, gw_tmp, _ = build_perfect_with_fragments(
                    s["train"][0], s["train"][1], indicates, normal_train, min_confirm=mc)
                va_t = s["val"][0]; va_ty = s["val"][1]
                r = evaluate_layer(va_t, va_ty, label, gk_tmp, gw_tmp, type_labels, threshold=th)
                if label == "fraud":
                    val_f_hits += r["hit"]; val_f_total += r["total"]
                else:
                    val_a_hits += r["hit"]; val_a_total += r["total"]
            # normal val 误报
            gk_comb = defaultdict(set); gw_comb = {}
            for label, s in [("fraud", fraud), ("ad", ad)]:
                gk_tmp, gw_tmp, _ = build_perfect_with_fragments(
                    s["train"][0], s["train"][1], indicates, normal_train, min_confirm=mc)
                for tn, ws in gk_tmp.items():
                    gk_comb[tn] |= ws
                gw_comb.update(gw_tmp)
            rn = evaluate_layer(norm["val"][0], norm["val"][1], "normal", gk_comb, gw_comb, type_labels, threshold=th)
            val_n_fp += rn["fp_count"]; val_n_total += rn["total"]

            val_score = (val_f_hits / val_f_total * 100 if val_f_total else 0) + \
                        (val_a_hits / val_a_total * 100 if val_a_total else 0)
            # 约束：normal 误报 <= 3，取 fraud+ad 召回和最高
            if val_n_fp <= 3:
                if best_g_cfg is None or val_score > best_g_cfg[0]:
                    best_g_cfg = (val_score, mc, th, val_f_hits / val_f_total * 100 if val_f_total else 0,
                                  val_a_hits / val_a_total * 100 if val_a_total else 0, val_n_fp)
    if best_g_cfg:
        _, g_mc, g_th, g_fr, g_ar, g_nfp = best_g_cfg
        print(f"  val 选参: min_confirm={g_mc}, threshold={g_th}, fraud={g_fr:.1f}%, ad={g_ar:.1f}%, normal误报={g_nfp}")
        g_res = {}
        for label, s in [("fraud", fraud), ("ad", ad)]:
            gk, gw, gs = build_perfect_with_fragments(
                s["train"][0], s["train"][1], indicates, normal_train, min_confirm=g_mc)
            te_t = s["test"][0]; te_ty = s["test"][1]
            tr = evaluate_layer(s["train"][0], s["train"][1], label, gk, gw, type_labels, threshold=g_th)
            r = evaluate_layer(te_t, te_ty, label, gk, gw, type_labels, threshold=g_th)
            g_res[label] = (gk, gw, gs, r)
            print(f"  [{label}] 完美词{gs['perfect_total']}+片段{gs['fragments_added']}={gs['total_words']}词")
            print(f"          (长词缺失{gs['long_missing']}, 确认片段{gs['fragments_confirmed']})")
            print(f"          train: 召回 {tr['recall']:.1f}%  test: 召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})  "
                  f"覆盖率 {r['type_coverage']}  准确率 {r['type_accuracy']:.1f}%")
            if r["zero_hit_types"]:
                print(f"          零命中: {r['zero_hit_types']}")
        g_kw = defaultdict(set); g_ww = {}
        for label_ in ["fraud", "ad"]:
            pk, pw, _, _ = g_res[label_]
            for tn, ws in pk.items():
                g_kw[tn] |= ws
            g_ww.update(pw)
        r = evaluate_layer(nte_t, nte_ty, "normal", g_kw, g_ww, type_labels, threshold=g_th, verbose=True)
        print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")
        if r["false_positives"]:
            for idx, txt, pl, bt, sc in r["false_positives"][:10]:
                print(f"      #{idx} '{txt}' -> {pl}({bt})")
    else:
        print("  val 上无满足 normal误报<=3 的配置")
        # 回退默认参数 min_confirm=2, threshold=THRESHOLD
        g_res = {}
        for label, s in [("fraud", fraud), ("ad", ad)]:
            gk, gw, gs = build_perfect_with_fragments(
                s["train"][0], s["train"][1], indicates, normal_train, min_confirm=2)
            te_t = s["test"][0]; te_ty = s["test"][1]
            r = evaluate_layer(te_t, te_ty, label, gk, gw, type_labels)
            g_res[label] = (gk, gw, gs, r)
        g_kw = defaultdict(set); g_ww = {}
        for label_ in ["fraud", "ad"]:
            pk, pw, _, _ = g_res[label_]
            for tn, ws in pk.items():
                g_kw[tn] |= ws
            g_ww.update(pw)
        g_th = THRESHOLD

    # ---- H: 混合分类器（整体最终） ----
    bar("H. 混合分类器 (TF-IDF+LR + 关键词救回 · 整体最终)")
    from hybrid_clf import HybridClassifier
    h_train_t = fraud["train"][0] + ad["train"][0] + norm["train"][0]
    h_train_l = ["fraud"] * len(fraud["train"][0]) + ["ad"] * len(ad["train"][0]) + ["normal"] * len(norm["train"][0])

    # val 选 C（约束：normal 误报 <= 4，取召回最高）
    best_cfg = None
    for C in [0.08, 0.1, 0.12, 0.15]:
        m = HybridClassifier(C=C, kw_min_score=3)
        m.fit(h_train_t, h_train_l, norm["train"][0], indicates, type_labels)
        nfp = sum(1 for t in norm["val"][0] if m.predict(t) != "normal")
        fh = sum(1 for t in fraud["val"][0] if m.predict(t) == "fraud")
        ah = sum(1 for t in ad["val"][0] if m.predict(t) == "ad")
        if nfp <= 4:
            score = fh / len(fraud["val"][0]) * 100 + ah / len(ad["val"][0]) * 100
            if best_cfg is None or score > best_cfg[0]:
                best_cfg = (score, C, fh / len(fraud["val"][0]) * 100, ah / len(ad["val"][0]) * 100, nfp)
    if best_cfg:
        _, C_best, fr_b, ar_b, nfp_b = best_cfg
        print(f"  val 选参: C={C_best}, fraud={fr_b:.1f}%, ad={ar_b:.1f}%, normal误报={nfp_b}")
        m_final = HybridClassifier(C=C_best, kw_min_score=3)
        m_final.fit(h_train_t, h_train_l, norm["train"][0], indicates, type_labels)
        # 独立 test 最终评测（严格隔离）
        f_h = sum(1 for t in fraud["test"][0] if m_final.predict(t) == "fraud")
        a_h = sum(1 for t in ad["test"][0] if m_final.predict(t) == "ad")
        n_fp = sum(1 for t in norm["test"][0] if m_final.predict(t) != "normal")
        print(f"  test 最终: fraud {f_h}/{len(fraud['test'][0])} = {f_h / len(fraud['test'][0]) * 100:.1f}%  "
              f"ad {a_h}/{len(ad['test'][0])} = {a_h / len(ad['test'][0]) * 100:.1f}%  "
              f"normal误报 {n_fp}/{len(norm['test'][0])}")
        h_fraud_recall = f_h / len(fraud['test'][0]) * 100
        h_ad_recall = a_h / len(ad['test'][0]) * 100
        h_normal_fp = n_fp
    else:
        print("  val 上无满足 normal误报<=4 的配置")
        h_fraud_recall, h_ad_recall, h_normal_fp = 0, 0, 0

    # =========== 汇总 ===========
    bar("七组评测汇总（A-E+G 关键词线 + H 混合分类器）")
    print(f"            {'A.模板词':>7} {'B.全量泄漏':>8} {'C.类型过滤':>8} {'D.严格隔离':>8} {'E.完美词':>8} {'G.类型片段':>8} {'H.混合分类':>8}")
    for label, s in [("fraud", fraud), ("ad", ad)]:
        at = s["train"][0] + s["val"][0] + s["test"][0]
        aty = s["train"][1] + s["val"][1] + s["test"][1]
        te_t = s["test"][0]
        te_ty = s["test"][1]
        ra = evaluate_layer(at, aty, label, pattern_kw, pattern_ww, type_labels)
        rb = evaluate_layer(at, aty, label, all_kw, all_ww, type_labels)
        skc, swc = build_type_filter_kw(s["train"][1], indicates)
        rc = evaluate_layer(te_t, te_ty, label, skc, swc, type_labels)
        skd, swd = build_strict_kw(s["train"][0], s["train"][1], indicates)
        rd = evaluate_layer(te_t, te_ty, label, skd, swd, type_labels)
        pk, pw, _ = perf[label]; re = evaluate_layer(te_t, te_ty, label, pk, pw, type_labels)
        _, _, _, rg = g_res[label]
        rh_val = h_fraud_recall if label == "fraud" else h_ad_recall
        print(f"  [{label}] 召回:    {ra['recall']:>5.1f}% {rb['recall']:>6.1f}% {rc['recall']:>6.1f}% {rd['recall']:>6.1f}% {re['recall']:>6.1f}% {rg['recall']:>6.1f}% {rh_val:>6.1f}%")
        print(f"         准确率:    {ra['type_accuracy']:>5.1f}% {rb['type_accuracy']:>6.1f}% {rc['type_accuracy']:>6.1f}% {rd['type_accuracy']:>6.1f}% {re['type_accuracy']:>6.1f}% {rg['type_accuracy']:>6.1f}%       -")
    rna = evaluate_layer(nall_t, nall_ty, "normal", pattern_kw, pattern_ww, type_labels)
    rnb = evaluate_layer(nall_t, nall_ty, "normal", all_kw, all_ww, type_labels)
    rnc = evaluate_layer(nte_t, nte_ty, "normal", c_kw, c_ww, type_labels)
    rnd = evaluate_layer(nte_t, nte_ty, "normal", s_kw, s_ww, type_labels)
    rne = evaluate_layer(nte_t, nte_ty, "normal", p_kw, p_ww, type_labels)
    rng = evaluate_layer(nte_t, nte_ty, "normal", g_kw, g_ww, type_labels, threshold=g_th)
    print(f"  [normal] 误报:    {rna['fp_count']:>3}/{rna['total']:<3}  {rnb['fp_count']:>3}/{rnb['total']:<3}   {rnc['fp_count']:>3}/{rnc['total']:<3}   {rnd['fp_count']:>3}/{rnd['total']:<3}   {rne['fp_count']:>3}/{rne['total']:<3}   {rng['fp_count']:>3}/{rng['total']:<3}   {h_normal_fp:>3}/{len(norm['test'][0])}")
    print(f"\n  A=人工基线 B=含泄漏 C=类型过滤 D=严格隔离 E=完美词 G=类型片段(关键词·test) H=混合分类器(整体·test)")
    print(f"  C/D/E/G/H 均为独立 test 评测 (严格隔离); A/B 为全量参考 (含泄漏)")


if __name__ == "__main__":
    main()
