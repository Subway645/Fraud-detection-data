import edge_tts
import asyncio
import os
import pandas as pd
from config import TEXT_DIR, FRAUD_DIR, AD_DIR

# ========== 配置区 ==========
CATEGORY = "both"  # "fraud" / "ad" / "both"
SEMAPHORE_LIMIT = 8
MIN_FILE_SIZE = 5000  # 小于5KB视为损坏
# ===========================

def is_valid_audio(filepath):
    """检查音频文件是否有效（通过文件大小判断）"""
    if not os.path.exists(filepath):
        return False
    size = os.path.getsize(filepath)
    if size < MIN_FILE_SIZE:
        return False
    return True

async def check_and_generate(csv_name, output_dir, prefix="", label=""):
    csv_path = os.path.join(TEXT_DIR, csv_name)
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="gbk")
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

    # 检查缺失或损坏的文件
    need_regenerate = []
    for i, row in df.iterrows():
        category = row["type"].replace("/", "_").replace(" ", "")
        filename = os.path.join(output_dir, f"{prefix}{i+1:03d}_{category}.mp3")
        if not is_valid_audio(filename):
            need_regenerate.append((i, row, filename))

    if not need_regenerate:
        print(f"{label}：所有 {len(df)} 条音频都有效（大小 > {MIN_FILE_SIZE//1024}KB），无需补全。")
        return

    print(f"{label}：共 {len(df)} 条，其中 {len(need_regenerate)} 条缺失或损坏（小于 {MIN_FILE_SIZE//1024}KB），开始重新生成...")

    async def generate_one(i, row, filename, retries=3):
        text = row["text"]
        voice = row["voice"]
        for attempt in range(retries):
            try:
                async with semaphore:
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(filename)
                    # 检查生成后的文件大小
                    if os.path.exists(filename) and os.path.getsize(filename) >= MIN_FILE_SIZE:
                        print(f"  ✅ 重新生成 [{i+1}/{len(df)}] {os.path.basename(filename)}")
                        return
                    else:
                        print(f"  ⚠️ [{i+1}/{len(df)}] 文件太小，可能损坏，重试中...")
                        if attempt < retries - 1:
                            await asyncio.sleep(2)
            except Exception as e:
                print(f"  ⚠️ [{i+1}/{len(df)}] 失败 (尝试 {attempt+1}/{retries})")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                else:
                    print(f"  ❌ [{i+1}/{len(df)}] 最终失败: {os.path.basename(filename)}")

    tasks = [generate_one(i, row, filename) for i, row, filename in need_regenerate]
    await asyncio.gather(*tasks)
    print(f"{label} 补全完成")

async def main():
    if CATEGORY in ["fraud", "both"]:
        await check_and_generate("诈骗话术.csv", FRAUD_DIR, prefix="", label="诈骗音频")
    if CATEGORY in ["ad", "both"]:
        await check_and_generate("电话广告话术.csv", AD_DIR, prefix="ad_", label="广告音频")

    print("所有补全任务完成")

if __name__ == "__main__":
    asyncio.run(main())