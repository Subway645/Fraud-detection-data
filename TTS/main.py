import edge_tts
import asyncio
import os
import pandas as pd

# 创建输出文件夹
os.makedirs("fraud_audio", exist_ok=True)

# 读取 CSV（用 gbk 编码）
df = pd.read_csv("诈骗话术_text_label_type_voice.csv", encoding="gbk")


async def generate_one(row, index):
    text = row["text"]
    category = row["type"]
    voice = row["voice"]

    safe_category = category.replace("/", "_").replace(" ", "")
    filename = f"fraud_audio/{index + 1:03d}_{safe_category}.mp3"

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)

    print(f"[{index + 1}/{len(df)}] {filename}")


async def main():
    print(f"共 {len(df)} 条话术，开始生成...")
    tasks = [generate_one(row, i) for i, row in df.iterrows()]
    await asyncio.gather(*tasks)
    print("全部完成！")


if __name__ == "__main__":
    asyncio.run(main())