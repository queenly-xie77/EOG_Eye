from flask import Flask, render_template, jsonify
import serial
import threading
import time
from collections import deque
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

app = Flask(__name__)

# =========================
# USER SETTINGS
# =========================
SERIAL_PORT = "/dev/tty.usbmodem1201"
BAUD_RATE = 115200
# =========================
# TUNED SETTINGS
# =========================
FS = 100
WINDOW_SECONDS = 15
MAX_SAMPLES = FS * WINDOW_SECONDS

# Blink-rate tuning based on medical averages (SBR)
BLINK_THRESHOLD_UV = 10.0      # Slightly increased to avoid noise/muscle jitter
BLINK_REFRACTORY_MS = 300      # Increased to 300ms (typical blink duration is 100-400ms)
BLINK_WINDOW_SEC = 15          # Expanded window to 15s for a more stable "average"

# Thresholds for a 15-second rolling window:
# Normal: ~3-4 blinks in 15s (12-16 per min)
# Drowsy: 6+ blinks in 15s (24+ per min)
# Extreme: 10+ blinks in 15s (40+ per min)
DROWSY_BLINK_COUNT = 10        
EXTREME_BLINK_COUNT = 30

# =========================
# GLOBAL DATA BUFFERS
# =========================
t_buf = deque(maxlen=MAX_SAMPLES)
raw_buf = deque(maxlen=MAX_SAMPLES)
filt_buf = deque(maxlen=MAX_SAMPLES)
activity_buf = deque(maxlen=MAX_SAMPLES)

status_data = {
    "state": "WAITING",
    "blink_count": 0,
    "last_update": 0
}

blink_events = deque()
blink_active = False
last_blink_time = -999.0

# =========================
# FILTER SETUP
# =========================
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

b_bp, a_bp = butter_bandpass(0.1, 8.0, FS, order=2)
zi_bp = lfilter_zi(b_bp, a_bp)
zi_state = zi_bp * 0.0

# =========================
# SERIAL
# =========================
ser = None

def connect_serial():
    global ser
    while ser is None:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
            time.sleep(2)
            print(f"Connected to {SERIAL_PORT}")
        except Exception as e:
            print("Waiting for Pico serial:", e)
            time.sleep(2)

def parse_line(line):
    try:
        parts = line.strip().split(",")
        if len(parts) != 2:
            return None
        t_ms = float(parts[0])
        uv = float(parts[1])
        return t_ms / 1000.0, uv
    except:
        return None

# =========================
# SIGNAL PROCESSING
# =========================
def process_sample(t, raw_uv):
    global zi_state, blink_active, last_blink_time

    # 1. Bandpass filter
    y, zi_state_new = lfilter(b_bp, a_bp, [raw_uv], zi=zi_state)
    zi_state = zi_state_new
    filt_uv = float(y[0])
    activity = abs(filt_uv)

    # 2. Update Buffers
    t_buf.append(t)
    raw_buf.append(raw_uv)
    filt_buf.append(filt_uv)
    activity_buf.append(activity)

    # 3. Blink Detection
    if activity > BLINK_THRESHOLD_UV:
        if (not blink_active) and ((t - last_blink_time) * 1000.0 > BLINK_REFRACTORY_MS):
            blink_events.append(t)
            last_blink_time = t
            blink_active = True
    else:
        blink_active = False

    # 4. Clean old blinks
    while blink_events and (t - blink_events[0]) > BLINK_WINDOW_SEC:
        blink_events.popleft()

    blink_count = len(blink_events)

    # 5. Logic
    if blink_count >= EXTREME_BLINK_COUNT:
        state = "EXTREME!!"
    elif blink_count >= DROWSY_BLINK_COUNT:
        state = "DROWSY  "
    else:
        state = "ALERT   "

    status_data.update({"state": state, "blink_count": blink_count, "last_update": time.time()})

    # 6. TERMINAL VISUALIZER
    # This creates a small progress bar showing the position in the 15s window
    window_pos = t % BLINK_WINDOW_SEC
    progress = int((window_pos / BLINK_WINDOW_SEC) * 20)
    bar = "█" * progress + "-" * (20 - progress)
    
    # Print status line to terminal without creating new lines
    print(f"\r[{bar}] Time: {t:7.2f}s | Window Pos: {window_pos:4.1f}s | Blinks: {blink_count:2} | State: {state}", end="")

def serial_reader():
    global ser

    connect_serial()

    while True:
        try:
            if ser is None:
                connect_serial()
                continue

            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            parsed = parse_line(line)
            if parsed is not None:
                t, raw_uv = parsed
                process_sample(t, raw_uv)

        except Exception as e:
            print("Serial error:", e)
            try:
                if ser:
                    ser.close()
            except:
                pass
            ser = None
            time.sleep(1)
            connect_serial()

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def data():
    return jsonify({
        "time": list(t_buf),
        "raw": list(raw_buf),
        "filtered": list(filt_buf),
        "activity": list(activity_buf),
        "state": status_data["state"],
        "blink_count": status_data["blink_count"]
    })

if __name__ == "__main__":
    thread = threading.Thread(target=serial_reader, daemon=True)
    thread.start()
    app.run(debug=False, host="127.0.0.1", port=5050)
    # Kill the port:
    # lsof -i :5000
    # kill -9 <PID>