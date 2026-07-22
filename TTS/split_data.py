import pandas as pd
import os
from sklearn.model_selection import train_test_split
from config import TEXT_DIR, SPLIT_OUTPUT_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO

def split_dataset(csv_name):
    """根据config中的配置划分数据集"""
    csv_path = os.path.join(TEXT_DIR, csv_name)
    df = pd.read_csv(csv_path, encoding="gbk")

    # 确定类别文件夹
    if "fraud" in csv_name:
        folder_name = "fraud"
    elif "ad" in csv_name:
        folder_name = "ad"
    else:
        folder_name = "other"

    output_dir = os.path.join(SPLIT_OUTPUT_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # 第一次划分
    train_val, test = train_test_split(
        df, test_size=TEST_RATIO, random_state=42, stratify=df["label"]
    )

    # 第二次划分
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val, test_size=val_ratio_adjusted, random_state=42, stratify=train_val["label"]
    )

    # 保存
    base_name = csv_name.replace(".csv", "")
    train.to_csv(os.path.join(output_dir, f"{base_name}_train.csv"), index=False, encoding="utf-8-sig")
    val.to_csv(os.path.join(output_dir, f"{base_name}_val.csv"), index=False, encoding="utf-8-sig")
    test.to_csv(os.path.join(output_dir, f"{base_name}_test.csv"), index=False, encoding="utf-8-sig")

    print(f"{csv_name} 划分完成 → {output_dir}")
    print(f"训练集: {len(train)} | 验证集: {len(val)} | 测试集: {len(test)}")

if __name__ == "__main__":
    split_dataset("fraud_utterances.csv")
    split_dataset("ad_utterances.csv")
    print("所有数据集划分完成！")