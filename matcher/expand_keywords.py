# -*- coding: utf-8 -*-
"""
净来电 · 关键词泛化引擎
========================
从 train 数据合法扩展关键词表，不偷看 val/test。

方法：
  1. 将 indicates 表中的长词(≥4字)拆成 2-gram/3-gram 子片段
  2. 在 train 同类文本中验证：如果子片段在 train 同类样本中命中 ≥2 次 → 纳入
  3. 权重继承：子片段从父词继承 strong/medium 权重
  4. 短词(≤3字)直接保留


"""
import csv
import os
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_DIR = os.path.join(BASE_DIR, "text_data")


def load_indicates():
    """从 relations.csv 加载 indicates 表"""
    word_types = defaultdict(dict)
    path = os.path.join(TEXT_DIR, "knowledge_graph", "relations.csv")
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["relation"] != "indicates":
                continue
            word = row["head_name"]
            type_name = row["tail_name"]
            note = row.get("note", "")
            weight = "medium"
            if "weight=strong" in note:
                weight = "strong"
            word_types[word][type_name] = weight
    return word_types


def load_splits(label):
    """返回 train, val, test 文本+类型"""
    result = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(TEXT_DIR, "splits", label, f"{label}_utterances_{split}.csv")
        texts, types = [], []
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                texts.append(row["text"])
                types.append(row["type"])
        result[split] = (texts, types)
    return result


def ngrams(text, n):
    """从文本提取所有 n-gram 子串"""
    if len(text) < n:
        return []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def expand_keywords(train_texts, train_types, indicates, min_confirm=2):
    """
    从 train 合法扩展关键词表。

    规则：
      1. ≤3 字的 indicates 词直接保留（短词泛化，train 里出现过就过）
      2. ≥4 字的 indicates 词拆成 2-gram/3-gram 子片段
      3. 子片段在 train 同类文本中命中 ≥ min_confirm 次 → 纳入
      4. 子片段继承父词权重

    返回: (keywords, word_weights)
    """
    train_types_set = set(train_types)

    # 按类型组织 train 文本
    type_to_texts = defaultdict(list)
    for text, t in zip(train_texts, train_types):
        type_to_texts[t].append(text)

    # starts with pattern JSON keywords (baseline, definitely in)
    # but we're building from indicates so we need the full word set
    train_keywords = defaultdict(set)
    train_word_weights = {}

    # ===== Step 1: Short words (≤3 chars) — keep if they appear in train =====
    short_words = {w for w in indicates if len(w) <= 3}
    short_in_train = set()
    for word in short_words:
        for text in train_texts:
            if word in text:
                short_in_train.add(word)
                break

    for word in short_in_train:
        for type_name, weight in indicates[word].items():
            if type_name in train_types_set:
                train_keywords[type_name].add(word)
                if word not in train_word_weights:
                    train_word_weights[word] = weight
                elif weight == "strong" and train_word_weights[word] == "medium":
                    train_word_weights[word] = "strong"

    # ===== Step 2: Long words (≥4 chars) → n-gram fragments =====
    long_words = [(w, indicates[w]) for w in indicates if len(w) >= 4]

    # For each long word, get all 2-grams and 3-grams
    fragment_candidates = defaultdict(list)  # fragment -> [(parent_word, type_name, weight)]
    for word, type_map in long_words:
        # 2-grams
        for n in [2, 3]:
            for frag in set(ngrams(word, n)):  # set = dedup within word
                for type_name, weight in type_map.items():
                    fragment_candidates[frag].append((word, type_name, weight))

    # Validate each fragment: must appear in ≥ min_confirm train texts of the SAME type
    expanded_count = 0
    for frag, parents in fragment_candidates.items():
        # Skip if already covered by short words
        if frag in short_in_train:
            continue

        # For each (type_name, weight) pair, check train confirmation
        confirmed_types = {}  # type_name -> best_weight
        for parent_word, type_name, weight in parents:
            if type_name not in train_types_set:
                continue
            if type_name not in type_to_texts:
                continue

            # Count occurrences in train texts of this type
            count = sum(1 for text in type_to_texts[type_name] if frag in text)
            if count >= min_confirm:
                if type_name not in confirmed_types:
                    confirmed_types[type_name] = weight
                elif weight == "strong" and confirmed_types[type_name] == "medium":
                    confirmed_types[type_name] = "strong"

        # Add confirmed fragments
        for type_name, weight in confirmed_types.items():
            train_keywords[type_name].add(frag)
            if frag not in train_word_weights:
                train_word_weights[frag] = weight
            elif weight == "strong" and train_word_weights[frag] == "medium":
                train_word_weights[frag] = "strong"
            expanded_count += 1

    return train_keywords, train_word_weights, {
        "short_kept": len(short_in_train),
        "short_total": len(short_words),
        "fragments_added": expanded_count,
        "total_keywords": len(train_word_weights),
        "type_count": len(train_keywords),
    }


def main():
    """自检：跑一遍扩展，看关键词增长了多少"""
    indicates = load_indicates()
    fraud = load_splits("fraud")
    ad = load_splits("ad")

    # Fraud expansion
    fkw, fww, fstats = expand_keywords(
        fraud["train"][0], fraud["train"][1], indicates, min_confirm=2
    )
    print("FRAUD 扩展统计:")
    for k, v in fstats.items():
        print(f"  {k}: {v}")

    # Ad expansion
    akw, aww, astats = expand_keywords(
        ad["train"][0], ad["train"][1], indicates, min_confirm=2
    )
    print("\nAD 扩展统计:")
    for k, v in astats.items():
        print(f"  {k}: {v}")

    # Show some new fragments
    # Compare with strict baseline
    train_vocab = set()
    for text in fraud["train"][0]:
        for word in indicates:
            if word in text:
                train_vocab.add(word)
    for text in ad["train"][0]:
        for word in indicates:
            if word in text:
                train_vocab.add(word)

    strict_count = len(train_vocab & set(indicates.keys()))
    expanded_fraud = len(fww)
    expanded_ad = len(aww)
    print(f"\n严格基线词数 (fraud+ad): {strict_count}")
    print(f"扩展后 fraud 词数: {expanded_fraud}")
    print(f"扩展后 ad 词数: {expanded_ad}")
    print(f"增长率: fraud={expanded_fraud/strict_count*100:.0f}% ad={expanded_ad/strict_count*100:.0f}%")


if __name__ == "__main__":
    main()
