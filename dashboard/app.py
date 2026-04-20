import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
import sqlite3
import re
import subprocess
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from config import settings

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

ALLOWED_AUDIO_EXT = ('.mp3', '.wav', '.ogg', '.m4a')


def get_db():
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    return render_template("index.html")


# ─── API: Events ─────────────────────────────────────────────────────────────

@app.route("/api/events")
def api_events():
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    cry_type = request.args.get("cry_type")

    conn = get_db()
    if cry_type:
        rows = conn.execute(
            "SELECT * FROM events WHERE cry_type = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (cry_type, limit, offset),
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM events WHERE cry_type = ?", (cry_type,)
        ).fetchone()[0]
    else:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()

    events = [dict(r) for r in rows]
    return jsonify({"events": events, "total": total})


# ─── API: Statistics ─────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    conn = get_db()

    distribution = conn.execute(
        "SELECT cry_type, COUNT(*) as count FROM events GROUP BY cry_type ORDER BY count DESC"
    ).fetchall()

    hourly = conn.execute(
        """SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
           FROM events
           WHERE timestamp >= datetime('now', 'localtime', '-24 hours')
           GROUP BY hour ORDER BY hour"""
    ).fetchall()

    daily = conn.execute(
        """SELECT date(timestamp) as day, COUNT(*) as count
           FROM events
           WHERE timestamp >= datetime('now', 'localtime', '-30 days')
           GROUP BY day ORDER BY day"""
    ).fetchall()

    daily_by_type = conn.execute(
        """SELECT date(timestamp) as day, cry_type, COUNT(*) as count
           FROM events
           WHERE timestamp >= datetime('now', 'localtime', '-7 days')
           GROUP BY day, cry_type ORDER BY day"""
    ).fetchall()

    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    today = conn.execute(
        "SELECT COUNT(*) FROM events WHERE date(timestamp) = date('now', 'localtime')"
    ).fetchone()[0]
    synced = conn.execute("SELECT COUNT(*) FROM events WHERE synced = 1").fetchone()[0]
    unsynced = conn.execute("SELECT COUNT(*) FROM events WHERE synced = 0").fetchone()[0]

    conn.close()

    return jsonify({
        "distribution": [dict(r) for r in distribution],
        "hourly": [dict(r) for r in hourly],
        "daily": [dict(r) for r in daily],
        "daily_by_type": [dict(r) for r in daily_by_type],
        "summary": {
            "total": total, "today": today,
            "synced": synced, "unsynced": unsynced,
        },
    })


# ─── API: Last Detection ─────────────────────────────────────────────────────

@app.route("/api/last-detection")
def api_last_detection():
    import json as _json
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM events ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    result = {"last": None, "all_probs": {}}
    if row:
        d = dict(row)
        try:
            d["all_probs"] = _json.loads(d.get("all_probs", "{}"))
        except Exception:
            d["all_probs"] = {}
        result["last"] = d
        result["all_probs"] = d["all_probs"]
    return jsonify(result)


# ─── API: Model Info ─────────────────────────────────────────────────────────

@app.route("/api/model-info")
def api_model_info():
    # model-info-dynamic-v1
    import json as _json_mod
    _state = os.path.join(settings.BASE_DIR, "storage", "active_model.txt")
    try:
        _active = open(_state).read().strip()
    except Exception:
        _active = "3cls"

    _mdir = os.path.join(settings.BASE_DIR, "model")
    _v3_tfl = os.path.join(_mdir, "cry_model_v3cls.tflite")
    _v3_meta = os.path.join(_mdir, "meta_final.json")
    _v5_tfl = os.path.join(_mdir, "cry_model_5cls_aug.tflite")
    _v5_meta = os.path.join(_mdir, "meta_5cls.json")

    if _active == "5cls" and all(os.path.exists(p) for p in [_v5_tfl, _v5_meta]):
        with open(_v5_meta) as _f:
            _m = _json_mod.load(_f)
        return jsonify({
            "active_model": "CNN 5-class (augmented)",
            "tflite_size_mb": round(os.path.getsize(_v5_tfl) / 1e6, 2),
            "classes": _m.get("classes", ["burping","hungry","no_cry","pain","tired"]),
            "test_accuracy": round(float(_m.get("test_acc", 0.896)) * 100, 1),
            "avg_inference_ms": 42,
            "input_shape": "(1, 64, 87, 1)",
            "feature_type": "Mel Spectrogram (64 bands, standardized)",
            "sample_rate": settings.SAMPLE_RATE,
            "duration_seconds": settings.DURATION,
            "confidence_threshold": settings.THRESHOLD,
            "n_params": _m.get("n_params"),
        })
    if _active == "3cls" and all(os.path.exists(p) for p in [_v3_tfl, _v3_meta]):
        with open(_v3_meta) as _f:
            _m = _json_mod.load(_f)
        return jsonify({
            "active_model": "CNN 3-class (augmented)",
            "tflite_size_mb": round(os.path.getsize(_v3_tfl) / 1e6, 2),
            "classes": _m.get("classes", ["hungry","no_cry","other_cry"]),
            "test_accuracy": round(float(_m.get("test_acc", _m.get("tflite_test_acc", 0.97))) * 100, 1),
            "avg_inference_ms": 37,
            "input_shape": "(1, 64, 87, 1)",
            "feature_type": "Mel Spectrogram (64 bands, standardized)",
            "sample_rate": settings.SAMPLE_RATE,
            "duration_seconds": settings.DURATION,
            "confidence_threshold": settings.THRESHOLD,
            "n_params": _m.get("n_params"),
        })
    # Fall through to legacy logic below

    v3cls_path = os.path.join(settings.BASE_DIR, "model", "cry_model_v3cls.tflite")
    v3cls_le_path = os.path.join(settings.BASE_DIR, "model", "label_encoder_3cls.joblib")
    v3cls_meta_path = os.path.join(settings.BASE_DIR, "model", "meta_final.json")
    yamnet_head_path = os.path.join(settings.BASE_DIR, "model", "yamnet_head.tflite")
    yamnet_base_path = os.path.join(settings.BASE_DIR, "model", "yamnet.tflite")
    cnn_path = os.path.join(settings.BASE_DIR, "model", "cry_model.tflite")
    le_path = os.path.join(settings.BASE_DIR, "model", "label_encoder.joblib")

    # 1. NEW: 3-class CNN trained on expanded dataset (no_cry / hungry / other_cry)
    if all(os.path.exists(p) for p in [v3cls_path, v3cls_le_path, v3cls_meta_path]):
        import json as _json_mod
        with open(v3cls_meta_path) as _f:
            _meta = _json_mod.load(_f)
        return jsonify({
            "active_model": "CNN v3cls (3-class)",
            "tflite_size_mb": round(os.path.getsize(v3cls_path) / 1e6, 2),
            "classes": _meta.get("classes", ["hungry", "no_cry", "other_cry"]),
            "test_accuracy": round(float(_meta.get("tflite_test_acc", 0.976)) * 100, 1),
            "avg_inference_ms": 86,
            "input_shape": "(1, 64, 87, 1)",
            "feature_type": "Mel Spectrogram (64 bands, standardized)",
            "sample_rate": settings.SAMPLE_RATE,
            "duration_seconds": settings.DURATION,
            "confidence_threshold": settings.THRESHOLD,
        })
    elif os.path.exists(yamnet_head_path) and os.path.exists(le_path):
        return jsonify({
            "active_model": "YAMNet + Classification Head",
            "yamnet_base_size_mb": round(os.path.getsize(yamnet_base_path) / 1e6, 2) if os.path.exists(yamnet_base_path) else None,
            "yamnet_head_size_mb": round(os.path.getsize(yamnet_head_path) / 1e6, 2),
            "classes": ["belly_pain", "discomfort", "hungry", "no_cry", "scared"],
            "test_accuracy": 98.7,
            "avg_inference_ms": 53,
            "embedding_dim": 3072,
            "feature_type": "YAMNet embeddings (avg+max+std pooling)",
            "sample_rate": settings.SAMPLE_RATE,
            "duration_seconds": settings.DURATION,
            "confidence_threshold": settings.THRESHOLD,
        })
    elif os.path.exists(cnn_path) and os.path.exists(le_path):
        return jsonify({
            "active_model": "CNN (TFLite) - Legacy",
            "tflite_size_mb": round(os.path.getsize(cnn_path) / 1e6, 2),
            "classes": ["belly_pain", "discomfort", "hungry", "scared"],
            "test_accuracy": 96.1,
            "avg_inference_ms": 11,
            "input_shape": "(1, 64, 87, 1)",
            "feature_type": "Mel Spectrogram (64 bands)",
            "sample_rate": settings.SAMPLE_RATE,
            "duration_seconds": settings.DURATION,
            "confidence_threshold": settings.THRESHOLD,
        })
    else:
        return jsonify({"active_model": "None"})



# ─── API: Controls ────────────────────────────────────────────────────────────

@app.route("/api/play-lullaby", methods=["POST"])
def api_play_lullaby():
    try:
        from actions.lullaby_player import play_random_lullaby
        play_random_lullaby()
        return jsonify({"status": "ok", "message": "Lullaby playback triggered"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/test-telegram", methods=["POST"])
def api_test_telegram():
    try:
        from actions.notifier import send_alert
        success = send_alert("test")
        return jsonify({"status": "ok" if success else "error"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify({
        "sample_rate": settings.SAMPLE_RATE,
        "duration": settings.DURATION,
        "channels": settings.CHANNELS,
        "threshold": settings.THRESHOLD,
        "telegram_configured": bool(settings.TELEGRAM_TOKEN and settings.CHAT_ID),
        "lullabies_count": len([
            f for f in os.listdir(settings.LULLABIES_DIR)
            if f.lower().endswith(ALLOWED_AUDIO_EXT)
        ]) if os.path.isdir(settings.LULLABIES_DIR) else 0,
    })


@app.route("/api/settings/threshold", methods=["POST"])
def api_set_threshold():
    try:
        data = request.get_json(force=True) or {}
        t = float(data.get("threshold"))
        if not 0.0 <= t <= 1.0:
            return jsonify({"status": "error", "message": "threshold must be between 0.0 and 1.0"}), 400
        cfg_path = os.path.join(settings.BASE_DIR, "config", "settings.py")
        with open(cfg_path, "r") as f:
            src = f.read()
        new_src, n = re.subn(r"^(THRESHOLD\s*=\s*).*$",
                             lambda m: m.group(1) + str(t),
                             src, count=1, flags=re.MULTILINE)
        if n == 0:
            return jsonify({"status": "error", "message": "THRESHOLD line not found in config"}), 500
        with open(cfg_path, "w") as f:
            f.write(new_src)
        settings.THRESHOLD = t
        try:
            subprocess.run(["sudo", "-n", "systemctl", "restart", "baby-cry.service"],
                           timeout=15, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        return jsonify({"status": "ok", "threshold": t})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/lullabies", methods=["GET"])
def api_lullabies_list():
    d = settings.LULLABIES_DIR
    if not os.path.isdir(d):
        return jsonify({"files": []})
    files = []
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(ALLOWED_AUDIO_EXT):
            continue
        p = os.path.join(d, f)
        try:
            size = os.path.getsize(p)
        except OSError:
            size = 0
        files.append({"name": f, "size_mb": round(size / 1024 / 1024, 2)})
    return jsonify({"files": files})


@app.route("/api/lullabies/upload", methods=["POST"])
def api_lullabies_upload():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "no file uploaded"}), 400
        up = request.files["file"]
        if not up.filename:
            return jsonify({"status": "error", "message": "empty filename"}), 400
        name = secure_filename(up.filename)
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_AUDIO_EXT:
            return jsonify({"status": "error",
                            "message": "allowed: " + ", ".join(ALLOWED_AUDIO_EXT)}), 400
        os.makedirs(settings.LULLABIES_DIR, exist_ok=True)
        dest = os.path.join(settings.LULLABIES_DIR, name)
        up.save(dest)
        return jsonify({"status": "ok", "name": name})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/lullabies/delete", methods=["POST"])
def api_lullabies_delete():
    try:
        data = request.get_json(force=True) or {}
        name = secure_filename(str(data.get("name", "")))
        if not name:
            return jsonify({"status": "error", "message": "no name"}), 400
        path = os.path.join(settings.LULLABIES_DIR, name)
        if not os.path.isfile(path):
            return jsonify({"status": "error", "message": "not found"}), 404
        os.remove(path)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        import numpy as np
        import librosa
        from model.classifier import predict
        from actions.notifier import send_alert

        dataset_base = os.path.join(settings.BASE_DIR, "donateacry_corpus")
        categories = [d for d in os.listdir(dataset_base)
                      if os.path.isdir(os.path.join(dataset_base, d))]
        if not categories:
            return jsonify({"status": "error", "message": "No dataset found"}), 400

        import random
        cat = random.choice(categories)
        cat_path = os.path.join(dataset_base, cat)
        files = [f for f in os.listdir(cat_path) if f.endswith(".wav")]
        if not files:
            return jsonify({"status": "error", "message": f"No wav files in {cat}"}), 400

        test_file = os.path.join(cat_path, random.choice(files))
        y, sr = librosa.load(test_file, sr=settings.SAMPLE_RATE, duration=settings.DURATION)
        if len(y) < settings.SAMPLE_RATE * settings.DURATION:
            y = np.pad(y, (0, settings.SAMPLE_RATE * settings.DURATION - len(y)))

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        cry_type, confidence, all_probs = predict(mel_spec_db)
        send_alert(cry_type, confidence=confidence, all_probs=all_probs, audio_path=test_file)

        # sim-class-mapping-v1
        _state_f = os.path.join(settings.BASE_DIR, "storage", "active_model.txt")
        try:
            _active = open(_state_f).read().strip()
        except Exception:
            _active = "3cls"
        if _active == "3cls":
            _MAP = {"belly_pain":"other_cry","burping":"other_cry","discomfort":"other_cry",
                    "tired":"other_cry","scared":"other_cry","lonely":"other_cry","cold_hot":"other_cry",
                    "hungry":"hungry","no_cry":"no_cry","other_cry":"other_cry"}
        else:
            _MAP = {"belly_pain":"pain","discomfort":"pain","burping":"burping",
                    "tired":"tired","hungry":"hungry","no_cry":"no_cry",
                    "scared":"pain","lonely":"pain","cold_hot":"pain"}
        cat = _MAP.get(cat, cat)
        if cry_type not in ("no_cry", "unknown"):                                                                                         
           from actions.lullaby_player import play_random_lullaby                                                                        
           play_random_lullaby()   
        return jsonify({
            "status": "ok",
            "actual_category": cat,
            "predicted": cry_type,
            "confidence": round(confidence, 4),
            "file": os.path.basename(test_file),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500




# ── Model switch (3cls / 5cls) ──────────────────────────────────────────────
import os as _os
_STATE = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "storage", "active_model.txt")

@app.route("/api/active", methods=["GET"])
def api_active():
    try:
        return jsonify({"active": open(_STATE).read().strip()})
    except Exception:
        return jsonify({"active": "3cls"})

@app.route("/api/switch/<which>", methods=["GET", "POST"])
def api_switch(which):
    if which not in ("3cls", "5cls"):
        return jsonify({"error": "must be 3cls or 5cls"}), 400
    _os.makedirs(_os.path.dirname(_STATE), exist_ok=True)
    with open(_STATE, "w") as f:
        f.write(which)
    return jsonify({"active": which, "ok": True})


# ── Lullaby & Speaker Controls ──────────────────────────────────────────────
import subprocess as _sp

def _find_usb_card():
    """Auto-detect USB speaker card number."""
    try:
        out = _sp.check_output(["aplay", "-l"], text=True)
        for line in out.splitlines():
            if "USB Audio" in line:
                return line.split(":")[0].replace("card ", "").strip()
    except Exception:
        pass
    return None 

@app.route("/api/volume", methods=["GET"])
def api_volume():
    try:
        card = _find_usb_card()
        if not card:
            return jsonify({"error": "USB speaker not found"}), 500
        out = _sp.check_output(["amixer", "-c", card, "sget", "PCM"], text=True)
        import re
        m = re.search(r'\[(\d+)%\]', out)
        return jsonify({"volume": int(m.group(1)) if m else 100})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/volume/<int:level>", methods=["POST"])
def api_set_volume(level):
    level = max(0, min(100, level))
    try:
        card = _find_usb_card()
        if not card:
            return jsonify({"error": "USB speaker not found"}), 500
        _sp.run(["amixer", "-c", card, "sset", "PCM", f"{level}%"], check=True, capture_output=True)
        return jsonify({"volume": level, "ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/lullaby/play", methods=["POST"])
def api_lullaby_play():
    try:
        from actions.lullaby_player import play_random_lullaby
        play_random_lullaby()
        return jsonify({"status": "playing"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/lullaby/stop", methods=["POST"])
def api_lullaby_stop():
    try:
        _sp.run(["pkill", "-f", "aplay"], capture_output=True)
        return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    from storage.event_store import init_db
    init_db()

    def _startup_notify():
        import requests, time as _t, os as _o
        # Skip if Flask reloader child process (prevents double notification)
        if _o.environ.get("WERKZEUG_RUN_MAIN") == "true":
            return
        # Skip if notified recently (prevents crash-loop spam)
        _flag = _o.path.join(settings.BASE_DIR, "storage", ".last_startup_notify")
        try:
            if _o.path.exists(_flag):
                age = _t.time() - _o.path.getmtime(_flag)
                if age < 60:  # cooldown: 60 seconds
                    return
            open(_flag, "w").write(str(_t.time()))
        except Exception:
            pass
        _t.sleep(3)
        try:
            _active = "3cls"
            try:
                import os as _o
                _active = open(_o.path.join(settings.BASE_DIR,"storage","active_model.txt")).read().strip()
            except Exception:
                pass
            _url = f"https://{settings.NGROK_DOMAIN}" if getattr(settings,"NGROK_DOMAIN","") else "http://raspberrypi.local:5555"
            _msg = f"\u2705 *Baby Cry Analyzer \u2014 Online*\n\n\U0001f4f1 Dashboard: {_url}\n\U0001f916 Model: *{_active}*\n\U0001f9ec ChatterBaby RF: *faol*\n\u23f0 Tizim ishga tushdi"
            _base = f"https://api.telegram.org/bot{settings.DASHBOARD_BOT_TOKEN}"
            for _cid in (getattr(settings,"CHAT_IDS",None) or [settings.CHAT_ID]):
                try:
                    requests.post(f"{_base}/sendMessage", json={"chat_id":_cid,"text":_msg,"parse_mode":"Markdown"}, timeout=10)
                except Exception:
                    pass
            print("[startup] notified", flush=True)
        except Exception as _e:
            print(f"[startup] failed: {_e}", flush=True)

    import threading as _th
    _th.Thread(target=_startup_notify, daemon=True).start()
    app.run(host="0.0.0.0", port=5555, debug=True)
