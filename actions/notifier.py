import requests
from config import settings
from storage import event_store

CRY_EMOJI = {
    "hungry": "🍼",
    "scared": "😨",
    "discomfort": "😣",
    "belly_pain": "🤢",
}


def _escape_md(text):
    """Escape underscores for Telegram Markdown."""
    return text.replace("_", "\\_")


def send_alert(cry_type, confidence=0.0, all_probs=None):
    emoji = CRY_EMOJI.get(cry_type, "❓")
    conf_pct = int(confidence * 100)

    prob_lines = ""
    if all_probs:
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            e = CRY_EMOJI.get(label, "❓")
            pct = int(prob * 100)
            arrow = "  <<" if label == cry_type else ""
            prob_lines += f"\n{e} {_escape_md(label)}: *{pct}%*{arrow}"

    message = (
        f"🚨 *Baby Cry Detected!*\n\n"
        f"{emoji} Type: *{_escape_md(cry_type)}*\n"
        f"📊 Confidence: *{conf_pct}%*\n"
        f"\n📋 *All Classes:*{prob_lines}\n\n"
        f"🎶 Playing Lullaby"
    )

    url = f"https://api.telegram.org/bot{settings.DASHBOARD_BOT_TOKEN}/sendMessage"
    # Broadcast to every configured chat; overall "sent" is True if at least one delivered.
    chat_ids = getattr(settings, "CHAT_IDS", None) or [settings.CHAT_ID]
    sent = False
    for cid in chat_ids:
        payload = {
            "chat_id": cid,
            "text": message,
            "parse_mode": "Markdown",
        }
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                sent = True
            else:
                print(f"Telegram error for chat {cid}: {r.status_code} {r.text[:200]}", flush=True)
        except Exception as e:
            print(f"Telegram send to {cid} failed: {e}", flush=True)

    event_store.log_event(cry_type, synced=sent, confidence=confidence, all_probs=all_probs)
    return sent
