import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from config import TEXT_DIR, FRAUD_DIR, AD_DIR, NORMAL_DIR

# ========== 配置区 ==========
CATEGORY = "ad"  # "fraud" / "ad" / "normal" / "all"
# ===========================

def add_reverb(y, sr, strength=0.5):
    """添加混响效果"""
    delay = int(0.08 * sr)
    decay = 0.3 + 0.5 * strength
    y_reverb = y.copy()
    delayed = np.zeros_like(y)
    delayed[delay:] = y[:-delay] * decay
    return y_reverb + delayed

def add_noise(y, level=0.005):
    """添加噪声"""
    noise = np.random.normal(0, level, y.shape)
    return y + noise

def apply_low_quality(y, sr, strength=0.5):
    """模拟低音质"""
    target_sr = int(16000 - 8000 * strength)
    if target_sr < 4000:
        target_sr = 4000
    y_low = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    y_restored = librosa.resample(y_low, orig_sr=target_sr, target_sr=sr)
    return y_restored

def adjust_volume(y, factor):
    """调整音量"""
    return y * factor

def adjust_pitch(y, sr, n_steps):
    """调整音调"""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def process_one(input_path, output_path, row):
    try:
        y, sr = librosa.load(input_path, sr=None)

        reverb_val = float(row.get("augment_reverb", 0))
        if reverb_val > 0:
            y = add_reverb(y, sr, reverb_val)

        noise_val = float(row.get("augment_noise", 0))
        if noise_val > 0:
            y = add_noise(y, noise_val)

        speed_val = float(row.get("augment_speed", 1.0))
        if speed_val != 1.0:
            y = librosa.effects.time_stretch(y, rate=speed_val)

        low_quality_val = float(row.get("augment_low_quality", 0))
        if low_quality_val > 0:
            y = apply_low_quality(y, sr, low_quality_val)

        volume_val = float(row.get("augment_volume", 1.0))
        if volume_val != 1.0:
            y = adjust_volume(y, volume_val)

        pitch_val = float(row.get("augment_pitch", 0))
        if pitch_val != 0:
            y = adjust_pitch(y, sr, pitch_val)

        sf.write(output_path, y, sr)
        return True
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return False

def process_series(csv_name, audio_dir, prefix="", label=""):
    csv_path = os.path.join(TEXT_DIR, csv_name)
    df = pd.read_csv(csv_path, encoding="gbk")

    has_augment = any(col.startswith("augment_") for col in df.columns)
    if not has_augment:
        print(f"⚠️ {label} 没有增强列，跳过")
        return

    output_dir = os.path.join(audio_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[{label}] 共 {len(df)} 条话术，开始后期处理...")

    for i, row in df.iterrows():
        category = row["type"].replace("/", "_").replace(" ", "")
        input_path = os.path.join(audio_dir, f"{prefix}{i+1:03d}_{category}.mp3")
        output_path = os.path.join(output_dir, f"{prefix}{i+1:03d}_{category}_processed.wav")

        if not os.path.exists(input_path):
            print(f"  ⚠️ 跳过 [{i+1}]：原始音频不存在")
            continue

        if process_one(input_path, output_path, row):
            print(f"  ✅ [{i+1}] 处理完成")

    print(f"[{label}] 处理完成！保存到 {os.path.basename(output_dir)}/")

if __name__ == "__main__":
    CATEGORIES = {
        "fraud": {"dir": FRAUD_DIR, "csv": "fraud_utterances.csv", "prefix": "", "label": "Fraud Audio"},
        "ad": {"dir": AD_DIR, "csv": "ad_utterances.csv", "prefix": "ad_", "label": "Ad Audio"},
        "normal": {"dir": NORMAL_DIR, "csv": "normal_utterances.csv", "prefix": "normal_", "label": "Normal Audio"},
    }

    for key, cfg in CATEGORIES.items():
        if CATEGORY in [key, "both", "all"]:
            process_series(cfg["csv"], cfg["dir"], prefix=cfg["prefix"], label=cfg["label"])
        elif CATEGORY == "both" and key == "normal":
            pass  # "both" only does fraud+ad for backward compat
        elif CATEGORY == "all":
            pass  # handled by "all" above

    print("\n所有后期处理任务完成！")