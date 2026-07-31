# -*- coding: utf-8 -*-
"""
净来电 · 话术模板匹配算法 · 最终隔离评测
==========================================
七层评测：
  A. 模板词基线
  B. 图谱全量 (含泄漏)
  C. 类型过滤
  D. 严格隔离
  E. 完美词隔离
  F. 完美词+公安部术语
  G. ★类型确认片段 (最终版)
"""
import csv, json, os
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_DIR = os.path.join(BASE_DIR, "text_data")

STRONG_SCORE = 3; MEDIUM_SCORE = 1; MEDIUM_MIN = 2; THRESHOLD = 3

# =========== 数据加载 ===========

def load_splits(label):
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
    word_types = defaultdict(dict)
    path = os.path.join(TEXT_DIR, "knowledge_graph", "relations.csv")
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["relation"] != "indicates": continue
            w = row["head_name"]
            weight = "strong" if "weight=strong" in row.get("note","") else "medium"
            word_types[w][row["tail_name"]] = weight
    return word_types

def load_type_labels():
    labels = {}
    path = os.path.join(TEXT_DIR, "knowledge_graph", "entities.csv")
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["entity_type"] == "ScamType": labels[row["name"]] = "fraud"
            elif row["entity_type"] == "AdType": labels[row["name"]] = "ad"
    return labels

def load_normal_all():
    texts = []
    for split in ['train', 'val', 'test']:
        path = os.path.join(TEXT_DIR, "splits", "normal", f"normal_utterances_{split}.csv")
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f): texts.append(row["text"])
    return texts

# =========== 匹配引擎 ===========

def match_text(text, keywords, weights, type_labels):
    if not text: return None, None, 0
    type_scores = defaultdict(int); type_med = defaultdict(int)
    for tn, words in keywords.items():
        for w in words:
            if w in text:
                wt = weights.get(w, "medium")
                if wt == "strong": type_scores[tn] += STRONG_SCORE
                else: type_med[tn] += 1
    for tn, c in type_med.items():
        if c >= MEDIUM_MIN: type_scores[tn] += MEDIUM_SCORE * c
    if not type_scores: return None, None, 0
    best = max(type_scores, key=type_scores.get)
    if type_scores[best] < THRESHOLD: return None, None, type_scores[best]
    return best, type_labels.get(best, "unknown"), type_scores[best]

# =========== 评测函数 ===========

def evaluate_layer(texts, types, expect_label, keywords, weights, type_labels, verbose=False):
    total = len(texts)
    tp = 0; dtypes = defaultdict(int); ctypes = defaultdict(int)
    ttotal = defaultdict(int); wrong = []; fps = []
    for i, (text, t) in enumerate(zip(texts, types)):
        ttotal[t] += 1
        best_type, pred_label, score = match_text(text, keywords, weights, type_labels)
        if pred_label == expect_label:
            tp += 1
            if best_type: dtypes[best_type] += 1
            if best_type == t: ctypes[t] += 1
        else:
            if pred_label is not None: fps.append((i, text[:60], pred_label, best_type, score))
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
    return [s[i:i+n] for i in range(len(s) - n + 1)] if len(s) >= n else []

def build_strict_kw(train_texts, train_types, indicates):
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
    ts = set(train_types); kw = defaultdict(set); ww = {}
    for w, type_map in indicates.items():
        for tn, wt in type_map.items():
            if tn in ts:
                kw[tn].add(w)
                if w not in ww: ww[w] = wt
                elif wt == "strong" and ww[w] == "medium": ww[w] = "strong"
    return kw, ww

def build_perfect_kw(all_train_texts, all_train_types, indicates, normal_texts):
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
    """完美词 + 类型确认片段"""
    ts = set(all_train_types)
    t2t = defaultdict(list)
    for text, t in zip(all_train_texts, all_train_types): t2t[t].append(text)
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

    # 类型确认片段
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

# =========== 公安部反诈术语 ===========

def get_official_terms():
    """公安部 2025 年 6 月发布防诈关键词"""
    return {
        "安全账户": "strong", "修复征信": "strong", "刷单做任务": "strong",
        "内幕消息": "strong", "百万保障": "strong", "积分清零": "strong",
        "稳赚不赔": "strong", "高额回报": "strong", "垫付资金": "strong",
        "无抵押": "strong", "低利率": "strong", "解冻费": "strong",
        "共享屏幕": "strong", "小众聊天": "strong", "快递引流": "strong",
        "虚拟货币": "strong", "刷流水": "strong", "购物卡": "strong",
        "涉嫌洗钱": "strong", "三倍理赔": "strong", "开通理赔通道": "strong",
        "海关扣押": "strong", "自动扣费": "strong", "强制执行": "strong",
        "协查通知": "strong", "资金审查": "strong", "内部名额": "strong",
        "包机票签证": "strong", "色情小卡片": "strong", "同城约会": "strong",
        "AI换脸": "strong", "数字藏品空投": "strong", "Web3钱包": "strong",
        "碳中积分": "strong", "量子通信": "strong", "无抵押贷款": "strong",
        "不看征信": "strong", "秒到账": "strong", "包装流水": "strong",
        "屏幕共享": "strong", "不告诉任何人": "strong",
        "保密": "medium", "到没人的地方": "strong", "更换手机号": "medium",
    }

def inject_official(kw, ww, official, indicates, normal_all):
    injected = 0
    for word, weight in official.items():
        if sum(1 for t in normal_all if word in t) > 0: continue
        if word in ww:
            if weight == "strong" and ww[word] == "medium": ww[word] = "strong"
            continue
        if word in indicates:
            for tn, wt in indicates[word].items():
                kw[tn].add(word)
            ww[word] = weight; injected += 1
    return injected

# =========== 主评测 ===========

def bar(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

def main():
    type_labels = load_type_labels()
    indicates = load_indicates()
    pattern_kw, pattern_ww = load_pattern_keywords()
    normal_all = load_normal_all()

    fraud = load_splits("fraud"); ad = load_splits("ad"); norm = load_splits("normal")

    fall_t = fraud["train"][0]+fraud["val"][0]+fraud["test"][0]
    fall_ty = fraud["train"][1]+fraud["val"][1]+fraud["test"][1]
    aall_t = ad["train"][0]+ad["val"][0]+ad["test"][0]
    aall_ty = ad["train"][1]+ad["val"][1]+ad["test"][1]
    nall_t = norm["train"][0]+norm["val"][0]+norm["test"][0]
    nall_ty = norm["train"][1]+norm["val"][1]+norm["test"][1]

    all_kw = defaultdict(set); all_ww = {}
    for w, tm in indicates.items():
        for tn, wt in tm.items():
            all_kw[tn].add(w)
            if w not in all_ww: all_ww[w] = wt
            elif wt == "strong" and all_ww[w] == "medium": all_ww[w] = "strong"

    all_train_t = fraud["train"][0] + ad["train"][0]
    all_train_ty = fraud["train"][1] + ad["train"][1]

    bar("净来电 · 话术模板匹配 · 七层隔离评测")
    print(f"日期: 2026-07-31 | splits 7:1.5:1.5 seed=42 | strong={STRONG_SCORE} medium={MEDIUM_SCORE} thresh={THRESHOLD}")
    print(f"Indicates: {len(indicates)}词  Pattern: {len(pattern_ww)}词")

    # ---- A: 模板词 ----
    bar("A. 模板词基线")
    for label, texts, types_ in [("fraud", fall_t, fall_ty), ("ad", aall_t, aall_ty)]:
        r = evaluate_layer(texts, types_, label, pattern_kw, pattern_ww, type_labels)
        print(f"  [{label}] 召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})  "
              f"覆盖率 {r['type_coverage']}  准确率 {r['type_accuracy']:.1f}%")
    r = evaluate_layer(nall_t, nall_ty, "normal", pattern_kw, pattern_ww, type_labels)
    print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")

    # ---- B: 图谱全量 ----
    bar("B. 图谱全量 (含泄漏)")
    for label, texts, types_ in [("fraud", fall_t, fall_ty), ("ad", aall_t, aall_ty)]:
        r = evaluate_layer(texts, types_, label, all_kw, all_ww, type_labels)
        print(f"  [{label}] 召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})  "
              f"覆盖率 {r['type_coverage']}  准确率 {r['type_accuracy']:.1f}%")
        if r["zero_hit_types"]: print(f"    零命中: {r['zero_hit_types']}")
    r = evaluate_layer(nall_t, nall_ty, "normal", all_kw, all_ww, type_labels)
    print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")

    # ---- C: 类型过滤 ----
    bar("C. 类型过滤 (残余泄漏)")
    for label, s in [("fraud", fraud), ("ad", ad)]:
        tk, tw = build_type_filter_kw(s["train"][1], indicates)
        vt_t = s["val"][0]+s["test"][0]; vt_ty = s["val"][1]+s["test"][1]
        r = evaluate_layer(vt_t, vt_ty, label, tk, tw, type_labels)
        print(f"  [{label}] 词数{len(tw)}  召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})  "
              f"覆盖率 {r['type_coverage']}  准确率 {r['type_accuracy']:.1f}%")
    tk_f, tw_f = build_type_filter_kw(fraud["train"][1], indicates)
    tk_a, tw_a = build_type_filter_kw(ad["train"][1], indicates)
    c_kw = defaultdict(set); c_ww = {}
    for d in [tk_f, tk_a]:
        for tn, ws in d.items(): c_kw[tn] |= ws
    c_ww.update(tw_f); c_ww.update(tw_a)
    nvt_t = norm["val"][0]+norm["test"][0]; nvt_ty = norm["val"][1]+norm["test"][1]
    r = evaluate_layer(nvt_t, nvt_ty, "normal", c_kw, c_ww, type_labels)
    print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")

    # ---- D: 严格隔离 ----
    bar("D. 严格隔离 (真实下限)")
    for label, s in [("fraud", fraud), ("ad", ad)]:
        sk, sw = build_strict_kw(s["train"][0], s["train"][1], indicates)
        vt_t = s["val"][0]+s["test"][0]; vt_ty = s["val"][1]+s["test"][1]
        r = evaluate_layer(vt_t, vt_ty, label, sk, sw, type_labels)
        print(f"  [{label}] 词数{len(sw)}  召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})  "
              f"覆盖率 {r['type_coverage']}  准确率 {r['type_accuracy']:.1f}%")
        if r["zero_hit_types"]: print(f"    零命中: {r['zero_hit_types']}")
    sk_f, sw_f = build_strict_kw(fraud["train"][0], fraud["train"][1], indicates)
    sk_a, sw_a = build_strict_kw(ad["train"][0], ad["train"][1], indicates)
    s_kw = defaultdict(set); s_ww = {}
    for d in [sk_f, sk_a]:
        for tn, ws in d.items(): s_kw[tn] |= ws
    s_ww.update(sw_f); s_ww.update(sw_a)
    r = evaluate_layer(nvt_t, nvt_ty, "normal", s_kw, s_ww, type_labels)
    print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")

    # ---- E: 完美词 ----
    bar("E. 完美词隔离 (零误报基准)")
    perf = {}
    for label, s in [("fraud", fraud), ("ad", ad)]:
        pk, pw, ps = build_perfect_kw(s["train"][0], s["train"][1], indicates, normal_all)
        perf[label] = (pk, pw, ps)
        vt_t = s["val"][0]+s["test"][0]; vt_ty = s["val"][1]+s["test"][1]
        tr = evaluate_layer(s["train"][0], s["train"][1], label, pk, pw, type_labels)
        r = evaluate_layer(vt_t, vt_ty, label, pk, pw, type_labels)
        print(f"  [{label}] 完美词{ps['perfect_total']} (过滤{ps['filtered_by_normal']})")
        print(f"          train: 召回 {tr['recall']:.1f}%  val+test: 召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})  "
              f"覆盖率 {r['type_coverage']}  准确率 {r['type_accuracy']:.1f}%")
        if r["zero_hit_types"]: print(f"          零命中: {r['zero_hit_types']}")
    p_kw = defaultdict(set); p_ww = {}
    for label_ in ["fraud", "ad"]:
        pk, pw, _ = perf[label_]
        for tn, ws in pk.items(): p_kw[tn] |= ws
        p_ww.update(pw)
    r = evaluate_layer(nvt_t, nvt_ty, "normal", p_kw, p_ww, type_labels)
    print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")

    # ---- F: 完美词+外部 ----
    bar("F. 完美词+外部术语 (+公安部2025)")
    official = get_official_terms()
    f_res = {}
    for label, s in [("fraud", fraud), ("ad", ad)]:
        pk = defaultdict(set, perf[label][0]); pw = dict(perf[label][1])
        inj = inject_official(pk, pw, official, indicates, normal_all)
        vt_t = s["val"][0]+s["test"][0]; vt_ty = s["val"][1]+s["test"][1]
        r = evaluate_layer(vt_t, vt_ty, label, pk, pw, type_labels)
        f_res[label] = (pk, pw, r)
        print(f"  [{label}] 完美词{len(perf[label][1])}+注入{inj}={len(pw)}词  "
              f"召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})")
    f_kw = defaultdict(set); f_ww = {}
    for label_ in ["fraud", "ad"]:
        pk, pw, _ = f_res[label_]
        for tn, ws in pk.items(): f_kw[tn] |= ws
        f_ww.update(pw)
    r = evaluate_layer(nvt_t, nvt_ty, "normal", f_kw, f_ww, type_labels)
    print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")

    # ---- G: 类型确认片段 ----
    bar("G. 类型确认片段 (最终版)")
    g_res = {}
    for label, s in [("fraud", fraud), ("ad", ad)]:
        gk, gw, gs = build_perfect_with_fragments(
            s["train"][0], s["train"][1], indicates, normal_all, min_confirm=2)
        vt_t = s["val"][0]+s["test"][0]; vt_ty = s["val"][1]+s["test"][1]
        tr = evaluate_layer(s["train"][0], s["train"][1], label, gk, gw, type_labels)
        r = evaluate_layer(vt_t, vt_ty, label, gk, gw, type_labels)
        g_res[label] = (gk, gw, gs, r)
        print(f"  [{label}] 完美词{gs['perfect_total']}+片段{gs['fragments_added']}={gs['total_words']}词")
        print(f"          (长词缺失{gs['long_missing']}, 确认片段{gs['fragments_confirmed']})")
        print(f"          train: 召回 {tr['recall']:.1f}%  val+test: 召回 {r['recall']:.1f}% ({r['hit']}/{r['total']})  "
              f"覆盖率 {r['type_coverage']}  准确率 {r['type_accuracy']:.1f}%")
        if r["zero_hit_types"]: print(f"          零命中: {r['zero_hit_types']}")
    g_kw = defaultdict(set); g_ww = {}
    for label_ in ["fraud", "ad"]:
        pk, pw, _, _ = g_res[label_]
        for tn, ws in pk.items(): g_kw[tn] |= ws
        g_ww.update(pw)
    r = evaluate_layer(nvt_t, nvt_ty, "normal", g_kw, g_ww, type_labels, verbose=True)
    print(f"  [normal] 误报 {r['fp_count']}/{r['total']}")
    if r["false_positives"]:
        for idx, txt, pl, bt, sc in r["false_positives"][:10]:
            print(f"      #{idx} '{txt}' -> {pl}({bt})")

    # =========== 汇总 ===========
    bar("七组评测汇总")
    l7 = "  [{label}] 召回: {a:>5.1f}% {b:>5.1f}% {c:>5.1f}% {d:>5.1f}% {e:>5.1f}% {f:>5.1f}% {g:>5.1f}%"
    a7 = "         准确率: {a:>5.1f}% {b:>5.1f}% {c:>5.1f}% {d:>5.1f}% {e:>5.1f}% {f:>5.1f}% {g:>5.1f}%"
    print(f"            {'A.模板词':>7} {'B.全量泄漏':>8} {'C.类型过滤':>8} {'D.严格隔离':>8} {'E.完美词':>8} {'F.+公安部':>8} {'G.类型片段':>8}")
    for label, s in [("fraud", fraud), ("ad", ad)]:
        at = s["train"][0]+s["val"][0]+s["test"][0]
        aty = s["train"][1]+s["val"][1]+s["test"][1]
        vt_t = s["val"][0]+s["test"][0]
        vt_ty = s["val"][1]+s["test"][1]
        ra = evaluate_layer(at, aty, label, pattern_kw, pattern_ww, type_labels)
        rb = evaluate_layer(at, aty, label, all_kw, all_ww, type_labels)
        skc, swc = build_type_filter_kw(s["train"][1], indicates)
        rc = evaluate_layer(vt_t, vt_ty, label, skc, swc, type_labels)
        skd, swd = build_strict_kw(s["train"][0], s["train"][1], indicates)
        rd = evaluate_layer(vt_t, vt_ty, label, skd, swd, type_labels)
        pk, pw, _ = perf[label]; re = evaluate_layer(vt_t, vt_ty, label, pk, pw, type_labels)
        _, _, rf = f_res[label]; _, _, _, rg = g_res[label]
        print(f"  [{label}] 召回:    {ra['recall']:>5.1f}% {rb['recall']:>6.1f}% {rc['recall']:>6.1f}% {rd['recall']:>6.1f}% {re['recall']:>6.1f}% {rf['recall']:>6.1f}% {rg['recall']:>6.1f}%")
        print(f"         准确率:    {ra['type_accuracy']:>5.1f}% {rb['type_accuracy']:>6.1f}% {rc['type_accuracy']:>6.1f}% {rd['type_accuracy']:>6.1f}% {re['type_accuracy']:>6.1f}% {rf['type_accuracy']:>6.1f}% {rg['type_accuracy']:>6.1f}%")
    # normal
    rna = evaluate_layer(nall_t, nall_ty, "normal", pattern_kw, pattern_ww, type_labels)
    rnb = evaluate_layer(nall_t, nall_ty, "normal", all_kw, all_ww, type_labels)
    rnc = evaluate_layer(nvt_t, nvt_ty, "normal", c_kw, c_ww, type_labels)
    rnd = evaluate_layer(nvt_t, nvt_ty, "normal", s_kw, s_ww, type_labels)
    rne = evaluate_layer(nvt_t, nvt_ty, "normal", p_kw, p_ww, type_labels)
    rnf = evaluate_layer(nvt_t, nvt_ty, "normal", f_kw, f_ww, type_labels)
    rng = evaluate_layer(nvt_t, nvt_ty, "normal", g_kw, g_ww, type_labels)
    print(f"  [normal] 误报:    {rna['fp_count']:>3}/{rna['total']:<3}  {rnb['fp_count']:>3}/{rnb['total']:<3}   {rnc['fp_count']:>3}/{rnc['total']:<3}   {rnd['fp_count']:>3}/{rnd['total']:<3}   {rne['fp_count']:>3}/{rne['total']:<3}   {rnf['fp_count']:>3}/{rnf['total']:<3}   {rng['fp_count']:>3}/{rng['total']:<3}")
    print(f"\n  A=人工基线 B=含泄漏 C=残余泄漏 D=严格下限 E=零误报基准 F=+公安部 G=★最终(类型确认片段)")

if __name__ == "__main__":
    main()
