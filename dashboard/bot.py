import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import sqlite3
import logging
import urllib.request
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    WebAppInfo, MenuButtonWebApp
)
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes
)
from config import settings as app_settings

# ── Config ────────────────────────────────────────────────────────────────────

BOT_TOKEN = "8133207647:AAEq5t8rCgkI_gBtN6-Hbr4DZ5RitLZs8U8"
WEBAPP_URL = ""  # set at startup from ngrok

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_webapp_url():
    """Get tunnel URL from ngrok local API or static domain setting."""
    if app_settings.NGROK_DOMAIN:
        return f"https://{app_settings.NGROK_DOMAIN}"
    try:
        req = urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=5)
        data = json.loads(req.read().decode())
        for tunnel in data.get("tunnels", []):
            if tunnel.get("public_url", "").startswith("https://"):
                return tunnel["public_url"]
    except Exception as e:
        logger.error(f"Failed to get ngrok URL: {e}")
    return None


def get_db():
    conn = sqlite3.connect(app_settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Uzbek translations ───────────────────────────────────────────────────────

CRY_EMOJI = {
    "hungry": "🍼",
    "scared": "😨",
    "discomfort": "😣",
    "belly_pain": "🤢",
    "no_cry": "🔇",
    "test": "🧪",
    "tired": "😴",
    "burping": "🫧",
    "other_cry": "😢",
    "pain": "🤕",
}

CRY_UZ = {
    "hungry": "Qorni och",
    "scared": "Qo'rqgan",
    "discomfort": "Noqulaylik",
    "belly_pain": "Qorni og'riyapti",
    "no_cry": "Yig'lamayapti",
    "test": "Test",
    "tired": "Charchagan",
    "burping": "Kekirish",
    "other_cry": "Boshqa yig'i",
    "pain": "Og'riq",
}


# ── Commands ──────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 Boshqaruv paneli", web_app=WebAppInfo(url=WEBAPP_URL))],
        [
            InlineKeyboardButton("📈 Statistika", callback_data="stats"),
            InlineKeyboardButton("📋 So'nggi hodisalar", callback_data="recent"),
        ],
        [
            InlineKeyboardButton("🧠 Model haqida", callback_data="model"),
            InlineKeyboardButton("🎵 Alla qo'yish", callback_data="lullaby"),
        ],
        [
            InlineKeyboardButton("🧪 Simulyatsiya", callback_data="simulate"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "👶 *Chaqaloq Yig'lash Analizatori*\n\n"
        "Xush kelibsiz! Men chaqaloqning yig'lashini real vaqtda kuzataman, "
        "sababini aniqlayman va avtomatik tinchlantiraman.\n\n"
        "*Boshqaruv paneli* ni bosib to'liq interfeyni oching, "
        "yoki quyidagi tezkor amallardan foydalaning.",
        parse_mode="Markdown",
        reply_markup=reply_markup,
    )


async def dashboard_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("📊 Boshqaruv paneli", web_app=WebAppInfo(url=WEBAPP_URL))]]
    await update.message.reply_text(
        "To'liq boshqaruv panelini ochish uchun bosing:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = _build_stats_text()
    await update.message.reply_text(text, parse_mode="Markdown")


async def recent_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = _build_recent_text()
    await update.message.reply_text(text, parse_mode="Markdown")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Chaqaloq Yig'lash Analizatori Boti*\n\n"
        "*Buyruqlar:*\n"
        "/start — Asosiy menyu\n"
        "/dashboard — Boshqaruv paneli\n"
        "/stats — Tezkor statistika\n"
        "/recent — So'nggi 10 ta hodisa\n"
        "/model — Model ma'lumotlari\n"
        "/simulate — Simulyatsiya ishga tushirish\n"
        "/lullaby — Alla qo'yish\n"
        "/help — Yordam ko'rsatish\n",
        parse_mode="Markdown",
    )


async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = _build_model_text()
    await update.message.reply_text(text, parse_mode="Markdown")


async def simulate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🧪 Simulyatsiya ishga tushirilmoqda...")
    result = _run_simulation()
    await msg.edit_text(result, parse_mode="Markdown")


async def lullaby_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = _play_lullaby()
    await update.message.reply_text(text, parse_mode="Markdown")


# ── Callback Query Handler ────────────────────────────────────────────────────

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "stats":
        text = _build_stats_text()
        await query.edit_message_text(text, parse_mode="Markdown")
    elif query.data == "recent":
        text = _build_recent_text()
        await query.edit_message_text(text, parse_mode="Markdown")
    elif query.data == "model":
        text = _build_model_text()
        await query.edit_message_text(text, parse_mode="Markdown")
    elif query.data == "lullaby":
        text = _play_lullaby()
        await query.edit_message_text(text, parse_mode="Markdown")
    elif query.data == "simulate":
        await query.edit_message_text("🧪 Simulyatsiya ishga tushirilmoqda...")
        result = _run_simulation()
        await query.edit_message_text(result, parse_mode="Markdown")

    # Add back button
    keyboard = [[InlineKeyboardButton("◀️ Orqaga", callback_data="menu")]]
    if query.data == "menu":
        keyboard = [
            [InlineKeyboardButton("📊 Boshqaruv paneli", web_app=WebAppInfo(url=WEBAPP_URL))],
            [
                InlineKeyboardButton("📈 Statistika", callback_data="stats"),
                InlineKeyboardButton("📋 So'nggi hodisalar", callback_data="recent"),
            ],
            [
                InlineKeyboardButton("🧠 Model haqida", callback_data="model"),
                InlineKeyboardButton("🎵 Alla qo'yish", callback_data="lullaby"),
            ],
            [InlineKeyboardButton("🧪 Simulyatsiya", callback_data="simulate")],
        ]
        await query.edit_message_text(
            "👶 *Chaqaloq Yig'lash Analizatori*\n\nAmalni tanlang:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
    elif query.data != "menu":
        try:
            await query.edit_message_reply_markup(
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception:
            pass


# ── Helper functions ──────────────────────────────────────────────────────────

def _escape_md(text):
    return text.replace("_", "\\_")


def _build_stats_text():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    today = conn.execute(
        "SELECT COUNT(*) FROM events WHERE date(timestamp) = date('now')"
    ).fetchone()[0]
    dist = conn.execute(
        "SELECT cry_type, COUNT(*) as cnt FROM events GROUP BY cry_type ORDER BY cnt DESC"
    ).fetchall()
    conn.close()

    lines = ["📈 *Tezkor Statistika*\n"]
    lines.append(f"Jami hodisalar: *{total}*")
    lines.append(f"Bugun: *{today}*\n")
    lines.append("*Taqsimot:*")
    for row in dist:
        emoji = CRY_EMOJI.get(row["cry_type"], "❓")
        uz = CRY_UZ.get(row["cry_type"], row["cry_type"])
        bar = "█" * min(row["cnt"], 20)
        lines.append(f"{emoji} {_escape_md(uz)}: *{row['cnt']}* {bar}")

    return "\n".join(lines)


def _build_recent_text():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM events ORDER BY timestamp DESC LIMIT 10"
    ).fetchall()
    conn.close()

    if not rows:
        return "📋 *So'nggi Hodisalar*\n\nHali hodisa qayd etilmagan."

    lines = ["📋 *So'nggi 10 ta Hodisa*\n"]
    for r in rows:
        emoji = CRY_EMOJI.get(r["cry_type"], "❓")
        uz = CRY_UZ.get(r["cry_type"], r["cry_type"])
        sync = "✅" if r["synced"] else "⏳"
        lines.append(f"{emoji} `{r['timestamp']}` — *{_escape_md(uz)}* {sync}")

    return "\n".join(lines)


def _build_model_text():
    tflite_path = os.path.join(app_settings.BASE_DIR, "model", "cry_model.tflite")
    size = round(os.path.getsize(tflite_path) / 1e6, 2) if os.path.exists(tflite_path) else "N/A"

    return (
        "🧠 *Model Ma'lumotlari*\n\n"
        f"*Faol model:* CNN (TFLite)\n"
        f"*Hajmi:* {size} MB\n"
        f"*Kirish:* Mel Spektrogramma (64 diapazon)\n"
        f"*Shakli:* (1, 64, 87, 1)\n"
        f"*Namuna tezligi:* {app_settings.SAMPLE_RATE} Hz\n"
        f"*Davomiyligi:* {app_settings.DURATION}s\n\n"
        f"*Sinflar:*\n"
        f"🍼 Qorni och | 😨 Qo'rqgan\n"
        f"😣 Noqulaylik | 🤢 Qorni og'riyapti | 🔇 Yig'lamayapti"
    )


def _play_lullaby():
    try:
        from actions.lullaby_player import play_random_lullaby
        play_random_lullaby()
        return "🎵 *Alla boshlandi*\n\nChaqaloqni tinchlantirish uchun alla aytilmoqda..."
    except Exception as e:
        return f"🎵 *Alla*\n\n⚠️ Ijro xatosi: {e}"


def _run_simulation():
    try:
        import numpy as np
        import librosa
        import random
        from model.classifier import predict
        from actions.notifier import send_alert

        dataset_base = os.path.join(app_settings.BASE_DIR, "donateacry_corpus")
        categories = [d for d in os.listdir(dataset_base)
                      if os.path.isdir(os.path.join(dataset_base, d))]

        if not categories:
            return "🧪 *Simulyatsiya Muvaffaqiyatsiz*\n\nMa'lumotlar to'plami topilmadi."

        cat = random.choice(categories)
        cat_path = os.path.join(dataset_base, cat)
        files = [f for f in os.listdir(cat_path) if f.endswith(".wav")]
        if not files:
            return f"🧪 *Simulyatsiya Muvaffaqiyatsiz*\n\n{cat} papkasida wav fayllar yo'q."

        test_file = os.path.join(cat_path, random.choice(files))
        y, sr = librosa.load(test_file, sr=app_settings.SAMPLE_RATE, duration=app_settings.DURATION)
        if len(y) < app_settings.SAMPLE_RATE * app_settings.DURATION:
            y = np.pad(y, (0, app_settings.SAMPLE_RATE * app_settings.DURATION - len(y)))

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        cry_type = predict(mel_spec_db)
        send_alert(cry_type)

        actual_uz = CRY_UZ.get(cat, cat)
        pred_uz = CRY_UZ.get(cry_type, cry_type)
        actual_emoji = CRY_EMOJI.get(cat, "❓")
        pred_emoji = CRY_EMOJI.get(cry_type, "❓")
        match = "✅" if cat == cry_type else "❌"

        return (
            f"🧪 *Simulyatsiya Natijasi*\n\n"
            f"📁 Fayl: `{os.path.basename(test_file)[:30]}...`\n"
            f"{actual_emoji} Haqiqiy: *{_escape_md(actual_uz)}*\n"
            f"{pred_emoji} Bashorat: *{_escape_md(pred_uz)}*\n"
            f"Natija: {match}\n\n"
            f"📱 Telegram xabarnoma yuborildi!"
        )
    except Exception as e:
        return f"🧪 *Simulyatsiya Muvaffaqiyatsiz*\n\n⚠️ Xato: {e}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global WEBAPP_URL

    url = get_webapp_url()
    if url:
        WEBAPP_URL = url
        logger.info(f"Dashboard URL: {WEBAPP_URL}")
    else:
        logger.error("Could not find ngrok tunnel URL! Start ngrok first.")
        logger.info("Run: ~/.local/bin/ngrok http 5555")
        return

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("dashboard", dashboard_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("recent", recent_cmd))
    app.add_handler(CommandHandler("model", model_cmd))
    app.add_handler(CommandHandler("simulate", simulate_cmd))
    app.add_handler(CommandHandler("lullaby", lullaby_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CallbackQueryHandler(button_handler))

    logger.info("Bot is starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    from storage.event_store import init_db
    init_db()
    main()
