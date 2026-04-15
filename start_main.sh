#!/bin/bash
# Start main.py in a detached tmux session.
# Rationale: portaudio on this Jetson requires a PTY to properly enumerate the USB mic
# in the presence of gdm-spawned pulseaudio. Running directly via nohup fails; tmux works.
tmux kill-session -t babycry 2>/dev/null
tmux new-session -d -s babycry "cd /home/jarvis/baby_cry && source venv/bin/activate && PULSE_SERVER=none LD_PRELOAD=/home/jarvis/baby_cry/venv/lib/python3.8/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0 python3 -u main.py 2>&1 | tee /tmp/baby_cry_listener.log"
echo "Started main.py in tmux session babycry"
echo "  attach: tmux attach -t babycry"
echo "  logs:   tail -f /tmp/baby_cry_listener.log"
echo "  stop:   tmux kill-session -t babycry"
