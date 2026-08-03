"""
净来电 · H 层混合分类方案
=========================
TF-IDF + 逻辑回归分类器 + 关键词强确认救回。

系统定位
--------
A-G 是关键词匹配线的逐层演进（G 为关键词最终，见 isolated_eval.py）；
H 层为整体方案的最终分类层，在严格隔离口径下召回最高
（fraud 80.0% / ad 85.3% / normal 误报 5/120）。

严格隔离流程
------------
  train 训练分类器 + 构建关键词
  val 选择超参（C、kw 救回分数）
  test 最终评测

判定规则
--------
  1. 分类器预测 fraud/ad → 采信
  2. 分类器预测 normal → 若关键词强命中（score >= kw_min_score）→ 改判
  3. 否则 → normal
"""
import csv
import json
import os
from collections import defaultdict

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_DIR = os.path.join(BASE_DIR, "text_data")


# =========== 数据加载 ===========

def load_split(label, split):
    """加载某类别指定划分的文本与子类型。"""
    texts, types_ = [], []
    with open(os.path.join(TEXT_DIR, "splits", label, f"{label}_utterances_{split}.csv"),
              encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            types_.append(row["type"])
    return texts, types_


def load_indicates():
    """加载 indicates 关系：word -> {type_name: weight}。"""
    indicates = defaultdict(dict)
    path = os.path.join(TEXT_DIR, "knowledge_graph", "relations.csv")
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["relation"] != "indicates":
                continue
            w = "strong" if "weight=strong" in row.get("note", "") else "medium"
            indicates[row["head_name"]][row["tail_name"]] = w
    return indicates


def load_type_labels():
    """加载类型实体到类别（fraud/ad）的映射。"""
    labels = {}
    path = os.path.join(TEXT_DIR, "knowledge_graph", "entities.csv")
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["entity_type"] == "ScamType":
                labels[row["name"]] = "fraud"
            elif row["entity_type"] == "AdType":
                labels[row["name"]] = "ad"
    return labels


def load_weak():
    """加载 pattern JSON 中的 weak 词（type -> [words]），用于平分类别时的语义取向。"""
    weak = defaultdict(list)
    for pfile in ["fraud_patterns.json", "ad_patterns.json"]:
        path = os.path.join(TEXT_DIR, "pattern_library", pfile)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for type_name, pat in data.get("patterns", {}).items():
            for kw in pat.get("keywords", []):
                if kw.get("weight") == "weak":
                    weak[type_name].append(kw["word"])
    return dict(weak)


_WEAK = None


def _get_weak():
    global _WEAK
    if _WEAK is None:
        _WEAK = load_weak()
    return _WEAK


# =========== 分词 ===========

def tokenize(text):
    """jieba 分词：过滤单字、纯数字与标点。"""
    toks = []
    for t in jieba.cut(text):
        t = t.strip()
        if len(t) >= 2 and not t.isdigit() and t not in "，。、！？：；,.!?:; ":
            toks.append(t)
    return toks


# =========== 关键词判定（严格构建） ===========

def kw_pred(text, kw, type_to_label):
    """关键词打分并判定类别，返回 (label, score)；未触发返回 (None, 0)。

    平分类别时按确定性规则决胜（weak 命中数 → 广告优先 → 类型名排序），
    与 matcher.py / isolated_eval.py 统一，消除跨进程哈希依赖。
    """
    ts = defaultdict(int)
    tm = defaultdict(int)
    for w, m in kw.items():
        if w in text:
            for tn, wt in m.items():
                if wt == "strong":
                    ts[tn] += 3
                else:
                    tm[tn] += 1
    for tn, c in tm.items():
        if c >= 2:
            ts[tn] += c
    if not ts:
        return None, 0

    max_score = max(ts.values())
    best_candidates = [tn for tn, sc in ts.items() if sc == max_score]
    if len(best_candidates) > 1:
        weak = _get_weak()
        wc = {tn: sum(1 for w in weak.get(tn, []) if w in text) for tn in best_candidates}
        max_weak = max(wc.values())
        top = [tn for tn in best_candidates if wc[tn] == max_weak] if max_weak > 0 else best_candidates
        ad_cands = [tn for tn in top if type_to_label.get(tn) == "ad"]
        best_tn = ad_cands[0] if ad_cands else sorted(top)[0]
    else:
        best_tn = best_candidates[0]
    return type_to_label.get(best_tn), ts[best_tn]


# =========== 混合分类器 ===========

class HybridClassifier:
    """TF-IDF + 逻辑回归分类器，normal 判定时以关键词强命中救回。"""

    def __init__(self, C=0.1, kw_min_score=3):
        self.C = C
        self.kw_min_score = kw_min_score
        self.vectorizer = None
        self.clf = None
        self.kw = None
        self.type_to_label = {}

    def fit(self, train_texts, train_labels, norm_train, indicates, type_to_label):
        """训练分类器并构建关键词表（关键词仅用 fraud/ad 的 train 文本）。"""
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2),
            sublinear_tf=True, max_features=8000)
        X = self.vectorizer.fit_transform(train_texts)
        self.clf = LogisticRegression(max_iter=5000, C=self.C, class_weight=None, random_state=42)
        self.clf.fit(X, train_labels)

        self.type_to_label = type_to_label
        all_texts = [t for t, l in zip(train_texts, train_labels) if l in ("fraud", "ad")]
        train_vocab = set()
        for text in all_texts:
            for w in indicates:
                if w in text:
                    train_vocab.add(w)
        kw = defaultdict(dict)
        for w in train_vocab:
            if any(w in nt for nt in norm_train):
                continue
            for tn, wt in indicates[w].items():
                kw[w][tn] = wt
        self.kw = kw

    def predict(self, text):
        """预测类别；分类器判 normal 时用关键词救回。"""
        c = self.clf.predict(self.vectorizer.transform([text]))[0]
        if c != "normal":
            return c
        k, sc = kw_pred(text, self.kw, self.type_to_label)
        if k is not None and sc >= self.kw_min_score:
            return k
        return "normal"

    def predict_many(self, texts):
        """批量预测。"""
        return [self.predict(t) for t in texts]


def main():
    print("=" * 64)
    print("净来电 · H 层混合分类方案 (分类器 + 关键词救回)")
    print("=" * 64)

    # 加载数据
    fraud_tr, _ = load_split("fraud", "train")
    fraud_va, _ = load_split("fraud", "val")
    fraud_te, _ = load_split("fraud", "test")
    ad_tr, _ = load_split("ad", "train")
    ad_va, _ = load_split("ad", "val")
    ad_te, _ = load_split("ad", "test")
    norm_tr = load_split("normal", "train")[0]
    norm_va = load_split("normal", "val")[0]
    norm_te = load_split("normal", "test")[0]

    indicates = load_indicates()
    type_to_label = load_type_labels()

    train_texts = fraud_tr + ad_tr + norm_tr
    train_labels = ["fraud"] * len(fraud_tr) + ["ad"] * len(ad_tr) + ["normal"] * len(norm_tr)

    print(f"train: fraud={len(fraud_tr)} ad={len(ad_tr)} normal={len(norm_tr)}")
    print(f"val:   fraud={len(fraud_va)} ad={len(ad_va)} normal={len(norm_va)}")
    print(f"test:  fraud={len(fraud_te)} ad={len(ad_te)} normal={len(norm_te)}")

    # val 上扫描 C（目标：normal 误报 <= 4 且召回最高）
    print("\nval 调参扫描 (目标: normal误报<=4 且召回最高):")
    best_cfg = None
    for C in [0.08, 0.1, 0.12, 0.15]:
        model = HybridClassifier(C=C, kw_min_score=3)
        model.fit(train_texts, train_labels, norm_tr, indicates, type_to_label)
        n_fp = sum(1 for t in norm_va if model.predict(t) != "normal")
        f_h = sum(1 for t in fraud_va if model.predict(t) == "fraud")
        a_h = sum(1 for t in ad_va if model.predict(t) == "ad")
        fr = f_h / len(fraud_va) * 100
        ar = a_h / len(ad_va) * 100
        print(f"  C={C}: fraud={fr:.1f}% ad={ar:.1f}% normal误报={n_fp}")
        if n_fp <= 4 and (best_cfg is None or fr + ar > best_cfg[0]):
            best_cfg = (fr + ar, C, fr, ar, n_fp)

    if not best_cfg:
        print("没有满足 normal误报<=4 的配置")
        return

    _, C_best, fr_b, ar_b, nfp_b = best_cfg
    print(f"\nval 最优: C={C_best}, fraud={fr_b:.1f}%, ad={ar_b:.1f}%, normal误报={nfp_b}")

    # test 最终评测
    model = HybridClassifier(C=C_best, kw_min_score=3)
    model.fit(train_texts, train_labels, norm_tr, indicates, type_to_label)
    f_h = sum(1 for t in fraud_te if model.predict(t) == "fraud")
    a_h = sum(1 for t in ad_te if model.predict(t) == "ad")
    n_fp = sum(1 for t in norm_te if model.predict(t) != "normal")
    print(f"\n=== TEST 最终 (train→val选参→test 严格隔离) ===")
    print(f"fraud 召回: {f_h}/{len(fraud_te)} = {f_h / len(fraud_te) * 100:.1f}%")
    print(f"ad 召回:    {a_h}/{len(ad_te)} = {a_h / len(ad_te) * 100:.1f}%")
    print(f"normal 误报: {n_fp}/{len(norm_te)} = {n_fp / len(norm_te) * 100:.1f}%")


if __name__ == "__main__":
    main()
