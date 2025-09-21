import cv2
import datetime
import threading
import requests
import subprocess
import time
import os
from ultralytics import YOLO

# ---------------- WiringOP (wPi) ----------------
WPI_PINS = [2, 5, 8, 11]   # 4 chân cần điều khiển
DEBOUNCE_SEC = 2.0         # thời gian giữ ON sau lần phát hiện gần nhất

def gpio_setup():
    for p in WPI_PINS:
        subprocess.run(["gpio", "mode", str(p), "out"], check=False)
        subprocess.run(["gpio", "write", str(p), "0"], check=False)

def gpio_all(value: int):
    # value: 1 = ON, 0 = OFF
    for p in WPI_PINS:
        subprocess.run(["gpio", "write", str(p), str(value)], check=False)

# ------------------------------------------------

# Load model
model = YOLO("best.pt")  # đổi sang file model của bạn

# Telegram config (khuyên dùng biến môi trường thay vì hardcode token)
BOT_TOKEN = os.getenv("BOT_TOKEN", "8169620330:AAEO_qBS1vXnA4Xb3jM4m2EOVQeWylhfzFU")
CHAT_ID = os.getenv("CHAT_ID", "-1002918477000")
TELEGRAM_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

last_alert = None
last_detect_time = None  # thời điểm phát hiện khói/lửa gần nhất

def alert(img, label):
    global last_alert

    if (last_alert is None) or (
        (datetime.datetime.utcnow() - last_alert).total_seconds() > 5
    ):
        last_alert = datetime.datetime.utcnow()
        filename = "alert.jpg"
        cv2.imwrite(filename, cv2.resize(img, dsize=None, fx=0.5, fy=0.5))

        thread = threading.Thread(target=send_telegram, args=(filename, label), daemon=True)
        thread.start()

def send_telegram(filename, label):
    try:
        with open(filename, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": CHAT_ID, "caption": f"🚨 {label} 🚨"}
            requests.post(TELEGRAM_URL, data=data, files=files, timeout=10)
        print("Send success:", label)
    except Exception as ex:
        print("Error sending telegram:", ex)

# Camera/video
cap = cv2.VideoCapture(
    r"C:\Users\Admin\OneDrive\Documents\A\Do_an_khoi_lua\Filming a Raging Forest Fire - North America - YouTube.mp4"
)

if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

# --- Setup GPIO ---
gpio_setup()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read video")
            break

        frame_resized = cv2.resize(frame, (800, 640))
        results = model(frame_resized)[0]

        detected = False

        if results.boxes:
            names = model.names  # dict {0:'Fire',1:'Smoke',...}
            labels = set()

            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]

                if "smoke" in cls_name.lower():
                    labels.add("BÁO KHÓI")
                elif "fire" in cls_name.lower():
                    labels.add("BÁO CHÁY")

            if labels:
                detected = True
                last_detect_time = time.time()
                frame_resized = results.plot()  # giữ bounding box
                # Bật tất cả 4 chân khi có phát hiện
                gpio_all(1)
                for lb in labels:
                    alert(frame_resized, lb)

        # Nếu không có phát hiện, sau DEBOUNCE_SEC sẽ tắt chân
        if not detected and last_detect_time is not None:
            if time.time() - last_detect_time > DEBOUNCE_SEC:
                gpio_all(0)

        cv2.imshow("Stream", frame_resized)
        if cv2.waitKey(1) == ord("q"):
            break

finally:
    # Tắt hết chân và giải phóng tài nguyên khi thoát
    gpio_all(0)
    cap.release()
    cv2.destroyAllWindows()
