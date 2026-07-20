import edge_tts
import asyncio
import pandas as pd
import os

# 要重新生成的序号
INDEX_TO_FIX = 225

df = pd.read_csv("诈骗话术_text_label_type_voice.csv", encoding="gbk")


async def fix_one():
    row = df.iloc[INDEX_TO_FIX - 1]
    text = row["text"]
    category = row["type"]
    voice = row["voice"]

    safe_category = category.replace("/", "_").replace(" ", "")
    filename = f"fraud_audio/{INDEX_TO_FIX:03d}_{safe_category}.mp3"

    print(f"🔧 重新生成第 {INDEX_TO_FIX} 条...")
    print(f"  话术: {text[:30]}...")
    print(f"  音色: {voice}")

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)

    print(f"已覆盖: {filename}")


if __name__ == "__main__":
    asyncio.run(fix_one())