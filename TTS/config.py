import os

BASE_DIR = r"C:\Users\Subway\Desktop\诈骗电话识别"
TEXT_DIR = os.path.join(BASE_DIR, "text_data")
AUDIO_DIR = os.path.join(BASE_DIR, "audio_data")
FRAUD_DIR = os.path.join(AUDIO_DIR, "fraud_audio")
AD_DIR = os.path.join(AUDIO_DIR, "ad_audio")