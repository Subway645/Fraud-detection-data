import edge_tts
import asyncio
import pandas as pd
import os
from config import TEXT_DIR, FRAUD_DIR, AD_DIR, NORMAL_DIR

# ===== 在这里修改 =====
CATEGORY = "normal"      # "fraud" / "ad" / "normal"
INDEX = 7
# =====================

CATEGORY_CONFIG = {
    "fraud":  {"csv": os.path.join(TEXT_DIR, "fraud_utterances.csv"),  "dir": FRAUD_DIR,  "prefix": ""},
    "ad":     {"csv": os.path.join(TEXT_DIR, "ad_utterances.csv"),     "dir": AD_DIR,     "prefix": "ad_"},
    "normal": {"csv": os.path.join(TEXT_DIR, "normal_utterances.csv"), "dir": NORMAL_DIR, "prefix": "normal_"},
}

cfg = CATEGORY_CONFIG[CATEGORY]
csv_file = cfg["csv"]
output_dir = cfg["dir"]
prefix = cfg["prefix"]

df = pd.read_csv(csv_file, encoding="gbk")

async def fix_one():
    row = df.iloc[INDEX - 1]
    text = row["text"]
    category = row["type"]
    voice = row["voice"]

    safe_category = category.replace("/", "_").replace(" ", "")
    filename = os.path.join(output_dir, f"{prefix}{INDEX:03d}_{safe_category}.mp3")

    print(f"重新生成第 {INDEX} 条（{CATEGORY}）")
    print(f"  话术: {text[:30]}...")
    print(f"  音色: {voice}")
    print(f"  保存到: {filename}")

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)
    print(f"✅ 已覆盖: {filename}")

if __name__ == "__main__":
    asyncio.run(fix_one())