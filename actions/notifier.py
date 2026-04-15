import requests
from config import settings
from storage import event_store

CRY_EMOJI = {
    "hungry": "🍼",
    "scared": "😨",
    "discomfort": "😣",
    "belly_pain": "🤢",
    "no_cry": "🔇",
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
    "tired": "Charchagan",
    "burping": "Kekirish",
    "other_cry": "Boshqa yig'i",
    "pain": "Og'riq",
}


def _escape_md(text):
    """Escape underscores for Telegram Markdown."""
    return text.replace("_", "\\_")


def send_alert(cry_type, confidence=0.0, all_probs=None, audio_path=None):
    emoji = CRY_EMOJI.get(cry_type, "❓")
    uz_name = CRY_UZ.get(cry_type, cry_type)
    conf_pct = int(confidence * 100)

    prob_lines = ""
    if all_probs:
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            e = CRY_EMOJI.get(label, "❓")
            uz = CRY_UZ.get(label, label)
            pct = int(prob * 100)
            arrow = "  <<" if label == cry_type else ""
            prob_lines += f"\n{e} {_escape_md(uz)}: *{pct}%*{arrow}"

    message = (
        f"🚨 *Chaqaloq yig'lashi aniqlandi!*\n\n"
        f"{emoji} Turi: *{_escape_md(uz_name)}*\n"
        f"📊 Aniqlik: *{conf_pct}%*\n"
        f"\n📋 *Barcha sinflar:*{prob_lines}\n\n"
        f"🎶 Alla aytilmoqda"
    )

    base_url = f"https://api.telegram.org/bot{settings.DASHBOARD_BOT_TOKEN}"
    chat_ids = getattr(settings, "CHAT_IDS", None) or [settings.CHAT_ID]
    sent = False

    for cid in chat_ids:
        try:
            if audio_path:
                # Send voice with alert text as caption (one message)
                with open(audio_path, "rb") as audio_file:
                    r = requests.post(
                        f"{base_url}/sendVoice",
                        data={
                            "chat_id": cid,
                            "caption": message,
                            "parse_mode": "Markdown",
                        },
                        files={"voice": ("cry.wav", audio_file, "audio/wav")},
                        timeout=30,
                    )
            else:
                # No audio — send text only
                r = requests.post(
                    f"{base_url}/sendMessage",
                    json={"chat_id": cid, "text": message, "parse_mode": "Markdown"},
                    timeout=10,
                )

            if r.status_code == 200:
                sent = True
            else:
                print(f"Telegram error for chat {cid}: {r.status_code} {r.text[:200]}", flush=True)
        except Exception as e:
            print(f"Telegram send to {cid} failed: {e}", flush=True)

    event_store.log_event(cry_type, synced=sent, confidence=confidence, all_probs=all_probs)
    return sent
