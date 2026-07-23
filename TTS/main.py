import edge_tts
import asyncio
import os
import pandas as pd
from config import TEXT_DIR, AUDIO_DIR

# ========== 配置区 ==========
CATEGORIES = {
    "fraud": {
        "csv": "fraud_utterances.csv",
        "dir": os.path.join(AUDIO_DIR, "fraud_audio"),
        "prefix": "",
        "label": "Fraud Audio"
    },
    "ad": {
        "csv": "ad_utterances.csv",
        "dir": os.path.join(AUDIO_DIR, "ad_audio"),
        "prefix": "ad_",
        "label": "Ad Audio"
    },
    "normal": {
        "csv": "normal_utterances.csv",
        "dir": os.path.join(AUDIO_DIR, "normal_audio"),
        "prefix": "normal_",
        "label": "Normal Audio"
    },
}

SELECTED = ["fraud", "ad"]  # 可选 "fraud", "ad", "normal"，或全选 ["fraud", "ad", "normal"]
SEMAPHORE_LIMIT = 10
# ===========================

async def generate_series(series_name, config, retries=3):
    csv_path = os.path.join(TEXT_DIR, config["csv"])
    output_dir = config["dir"]
    prefix = config["prefix"]
    label = config["label"]

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="gbk")
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

    async def generate_one(row, index):
        text = row["text"]
        category = row["type"]
        voice = row["voice"]
        safe_category = category.replace("/", "_").replace(" ", "")
        filename = os.path.join(output_dir, f"{prefix}{index+1:03d}_{safe_category}.mp3")

        for attempt in range(retries):
            try:
                async with semaphore:
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(filename)
                    print(f"✅ [{index+1}/{len(df)}] {os.path.basename(filename)}")
                    return
            except Exception as e:
                print(f"⚠️ 第 {index+1} 条失败 (尝试 {attempt+1}/{retries})")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                else:
                    print(f"❌ 第 {index+1} 条最终失败: {os.path.basename(filename)}")

    print(f"\n[{label}] 共 {len(df)} 条，开始生成到 {os.path.basename(output_dir)}...")
    tasks = [generate_one(row, i) for i, row in df.iterrows()]
    await asyncio.gather(*tasks)
    print(f"[{label}] 生成完成！")

async def main():
    if not SELECTED:
        print("⚠️ 请先在 SELECTED 中选择要生成的系列（如 ['fraud', 'ad']）")
        return

    for series_name in SELECTED:
        if series_name not in CATEGORIES:
            print(f"⚠️ 未知系列 '{series_name}'，跳过")
            continue
        await generate_series(series_name, CATEGORIES[series_name])

    print("\n所有任务完成！")

if __name__ == "__main__":
    asyncio.run(main())