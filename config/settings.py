import os

# Telegram Settings — alert bot (original @xv_jarvis_bot)
TELEGRAM_TOKEN = "8405620210:AAFRmrdrUScuHe1kGNGqH-99yu2Q3GN9-ZM"
# Primary chat (used by legacy code and as CHAT_IDS[0])
CHAT_ID = "752030660"
# All chats that receive baby-cry alerts
CHAT_IDS = ["752030660", "1444766498"]

# Telegram Settings — dashboard bot (@nm_clawd_bot)
DASHBOARD_BOT_TOKEN = "8133207647:AAEq5t8rCgkI_gBtN6-Hbr4DZ5RitLZs8U8"

# Audio Settings
SAMPLE_RATE = 22050
DURATION = 2  # seconds
CHANNELS = 1
THRESHOLD = 0.7
AUDIO_DEVICE = "default"  # ALSA/PortAudio default on Raspberry Pi
AUDIO_OUTPUT_DEVICE = 25  # USB Speaker Phone

# Ngrok tunnel settings
# Set NGROK_DOMAIN to your claimed static domain (e.g. "my-app.ngrok-free.app")
# Leave empty to use dynamic URL (read from ngrok local API)
NGROK_DOMAIN = "cherrie-uninterlocked-superinsistently.ngrok-free.dev"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'cry_model.tflite')
LULLABIES_DIR = os.path.join(BASE_DIR, 'lullabies')
DB_PATH = os.path.join(BASE_DIR, 'storage', 'events.db')
