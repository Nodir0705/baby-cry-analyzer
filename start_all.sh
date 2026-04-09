#!/bin/bash
cd /home/jarvis/baby_cry
source venv/bin/activate

echo "=== Baby Cry Analyzer ==="

# Kill old processes
killall -q python3 ngrok cloudflared 2>/dev/null
sleep 1

# 1. Flask dashboard
echo "[1/4] Dashboard..."
nohup bash -c 'cd /home/jarvis/baby_cry && source venv/bin/activate && exec python3 dashboard/app.py' > /tmp/dashboard.log 2>&1 &
sleep 2

# 2. Ngrok tunnel
echo "[2/4] Tunnel (ngrok)..."
NGROK_DOMAIN=$(python3 -c "from config.settings import NGROK_DOMAIN; print(NGROK_DOMAIN)" 2>/dev/null)
if [ -n "$NGROK_DOMAIN" ]; then
    nohup bash -c "exec /home/jarvis/.local/bin/ngrok http 5555 --domain=$NGROK_DOMAIN" > /tmp/ngrok.log 2>&1 &
else
    nohup bash -c 'exec /home/jarvis/.local/bin/ngrok http 5555' > /tmp/ngrok.log 2>&1 &
fi
sleep 5

# Get tunnel URL from ngrok local API
TUNNEL_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys,json; d=json.load(sys.stdin); print(next(t['public_url'] for t in d['tunnels'] if t['public_url'].startswith('https')))" 2>/dev/null)
if [ -z "$TUNNEL_URL" ]; then
    echo "ERROR: Tunnel failed"; exit 1
fi
echo "   URL: $TUNNEL_URL"

# 3. Telegram bot
echo "[3/4] Bot..."
nohup bash -c 'cd /home/jarvis/baby_cry && source venv/bin/activate && exec python3 dashboard/bot.py' > /tmp/bot.log 2>&1 &
sleep 2

# 4. Real-time listener
echo "[4/4] Listener..."
nohup bash -c 'cd /home/jarvis/baby_cry && source venv/bin/activate && exec python3 -u main.py' > /tmp/baby_cry_listener.log 2>&1 &
sleep 3

echo ""
echo "=== All running ==="
echo "Dashboard : $TUNNEL_URL"
echo "Listener  : USB mic (device 24)"
echo ""
echo "Processes:"
ps -eo pid,comm,args --no-headers | grep -E 'python3|ngrok' | grep -v grep
