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
FS = 100
WINDOW_SECONDS = 15
MAX_SAMPLES = FS * WINDOW_SECONDS

# Fatigue tuning
BLINK_THRESHOLD_UV = 8.0
LONG_CLOSURE_MS = 350
FATIGUE_EVENT_COUNT = 4
FATIGUE_WINDOW_SEC = 20

# =========================
# GLOBAL DATA BUFFERS
# =========================
t_buf = deque(maxlen=MAX_SAMPLES)
raw_buf = deque(maxlen=MAX_SAMPLES)
filt_buf = deque(maxlen=MAX_SAMPLES)
activity_buf = deque(maxlen=MAX_SAMPLES)

status_data = {
    "state": "WAITING",
    "long_closures": 0,
    "last_update": 0
}

long_closure_events = deque()
eye_closed = False
eye_close_start = None

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
    global zi_state, eye_closed, eye_close_start

    # Bandpass filter
    y, zi_state_new = lfilter(b_bp, a_bp, [raw_uv], zi=zi_state)
    zi_state = zi_state_new
    filt_uv = float(y[0])

    activity = abs(filt_uv)

    t_buf.append(t)
    raw_buf.append(raw_uv)
    filt_buf.append(filt_uv)
    activity_buf.append(activity)

    # Eye closure logic
    if activity > BLINK_THRESHOLD_UV:
        if not eye_closed:
            eye_closed = True
            eye_close_start = t
    else:
        if eye_closed:
            duration_ms = (t - eye_close_start) * 1000.0
            if duration_ms >= LONG_CLOSURE_MS:
                long_closure_events.append(t)
            eye_closed = False
            eye_close_start = None

    # Remove old events
    while long_closure_events and (t - long_closure_events[0]) > FATIGUE_WINDOW_SEC:
        long_closure_events.popleft()

    long_count = len(long_closure_events)

    if long_count >= FATIGUE_EVENT_COUNT:
        state = "FATIGUE WARNING"
    elif long_count >= 2:
        state = "DROWSY"
    else:
        state = "ALERT"

    status_data["state"] = state
    status_data["long_closures"] = long_count
    status_data["last_update"] = time.time()

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
        "long_closures": status_data["long_closures"]
    })

if __name__ == "__main__":
    thread = threading.Thread(target=serial_reader, daemon=True)
    thread.start()
    app.run(debug=False, host="127.0.0.1", port=5060)