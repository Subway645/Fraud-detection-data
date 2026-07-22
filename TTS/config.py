import os

BASE_DIR = r"C:\Users\Subway\Desktop\诈骗电话识别"

# ========== 数据目录 ==========
TEXT_DIR = os.path.join(BASE_DIR, "text_data")
AUDIO_DIR = os.path.join(BASE_DIR, "audio_data")
FRAUD_DIR = os.path.join(AUDIO_DIR, "fraud_audio")
AD_DIR = os.path.join(AUDIO_DIR, "ad_audio")
# ========== 数据划分配置 ==========
SPLIT_OUTPUT_DIR = os.path.join(TEXT_DIR, "splits")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15