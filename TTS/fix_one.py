import edge_tts
import asyncio
import pandas as pd
import os
from config import TEXT_DIR, FRAUD_DIR, AD_DIR

# ===== 在这里修改你要修复的内容 =====
CATEGORY = "ad"      # 可选 "fraud" 或 "ad"
INDEX = 1            # 要修复的序号（从1开始）
# ===================================

# 根据类别选择 CSV 文件和输出目录
if CATEGORY == "fraud":
    csv_file = os.path.join(TEXT_DIR, "诈骗话术.csv")
    output_dir = FRAUD_DIR
    prefix = ""
else:
    csv_file = os.path.join(TEXT_DIR, "电话广告话术.csv")
    output_dir = AD_DIR
    prefix = "ad_"

df = pd.read_csv(csv_file, encoding="gbk")

async def fix_one():
    row = df.iloc[INDEX - 1]
    text = row["text"]
    category = row["type"]
    voice = row["voice"]

    safe_category = category.replace("/", "_").replace(" ", "")
    filename = os.path.join(output_dir, f"{prefix}{INDEX:03d}_{safe_category}.mp3")

    print(f"🔧 重新生成第 {INDEX} 条（{CATEGORY}）")
    print(f"  话术: {text[:30]}...")
    print(f"  音色: {voice}")
    print(f"  保存到: {filename}")

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)
    print(f"✅ 已覆盖: {filename}")

if __name__ == "__main__":
    asyncio.run(fix_one())