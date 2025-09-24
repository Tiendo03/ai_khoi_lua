import cv2, time, os, sys, threading, subprocess, datetime, signal, requests
from ultralytics import YOLO
import numpy as np

VIDEO_SRC    = os.getenv("VIDEO_SRC", "/dev/video0")  
MODEL_PATH   = os.getenv("MODEL_PATH", "best.pt")
WPI_PINS     = [2, 5, 8, 11]
DEBOUNCE_SEC = float(os.getenv("DEBOUNCE_SEC", "2.0"))

IMGSZ        = int(os.getenv("IMGSZ", "416"))         
CONF         = float(os.getenv("CONF", "0.35"))
IOU          = float(os.getenv("IOU", "0.45"))
MAX_DET      = int(os.getenv("MAX_DET", "20"))

# Telegram
BOT_TOKEN    = os.getenv("BOT_TOKEN", "8169620330:AAEO_qBS1vXnA4Xb3jM4m2EOVQeWylhfzFU")
CHAT_ID      = os.getenv("CHAT_ID", "-1002918477000")
TELEGRAM_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto" if (BOT_TOKEN and CHAT_ID) else None
ALERT_INTERVAL_SEC = 5  
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "82")) 
SEND_MAX_W   = int(os.getenv("SEND_MAX_W", "640"))    

def gpio_setup():
    for p in WPI_PINS:
        subprocess.run(["gpio", "mode", str(p), "out"], check=False)
        subprocess.run(["gpio", "write", str(p), "0"], check=False)

def gpio_all(v: int):
    for p in WPI_PINS:
        subprocess.run(["gpio", "write", str(p), str(v)], check=False)

def send_telegram_ndarray(img_bgr, label):
    if not TELEGRAM_URL:
        print("Telegram OFF (thiếu BOT_TOKEN/CHAT_ID).")
        return
    try:
        h, w = img_bgr.shape[:2]
        if w > SEND_MAX_W:
            scale = SEND_MAX_W / float(w)
            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            print("Không encode được JPEG.")
            return
        files = {"photo": ("alert.jpg", buf.tobytes(), "image/jpeg")}
        data = {"chat_id": CHAT_ID, "caption": f"🚨 {label} 🚨"}
        requests.post(TELEGRAM_URL, data=data, files=files, timeout=8)
        print("Đã gửi Telegram:", label)
    except Exception as ex:
        print("Lỗi gửi Telegram:", ex)

running = True
def request_stop(*_):
    global running
    running = False
    print(">> Dừng... tắt GPIO an toàn.")

signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)

def stdin_q_listener():
    for line in sys.stdin:
        if line.strip().lower() == "q":
            request_stop()
            break
threading.Thread(target=stdin_q_listener, daemon=True).start()

def open_capture(src):
    return cv2.VideoCapture(src)

def main():
    global running
    model = YOLO(MODEL_PATH)

    # Tìm id class 'fire' và 'smoke' để filter ngay trong model
    names = model.names if hasattr(model, "names") else {}
    classes_keep = []
    for k, v in names.items():
        vl = str(v).lower()
        if "fire" in vl or "smoke" in vl:
            classes_keep.append(int(k))

    cap = open_capture(VIDEO_SRC)
    if not cap.isOpened():
        print("Không mở được nguồn video:", VIDEO_SRC)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
    cap.set(cv2.CAP_PROP_FPS,          15)

    gpio_setup()
    print(f"Đang giám sát (headless). Ctrl+C hoặc gõ 'q'+Enter để thoát. imgsz={IMGSZ}")

    _ = model.predict(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), imgsz=IMGSZ, conf=CONF, iou=IOU,
                      max_det=MAX_DET, classes=classes_keep or None, verbose=False, device="cpu")

    last_detect_time = None
    last_alert_ts = 0

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_in = cv2.resize(frame, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)

            res = model.predict(
                frame_in,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                max_det=MAX_DET,
                classes=classes_keep or None,
                verbose=False,
                device="cpu"   
            )[0]

            detected = False
            labels = set()
            if res.boxes and len(res.boxes) > 0:
                n = model.names
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    name = n.get(cls_id, "").lower()
                    if "smoke" in name:
                        labels.add("BÁO KHÓI")
                    elif "fire" in name:
                        labels.add("BÁO CHÁY")

            if labels:
                detected = True
                last_detect_time = time.time()
                gpio_all(1)
                vis = res.plot()

                if time.time() - last_alert_ts > ALERT_INTERVAL_SEC:
                    last_alert_ts = time.time()
                    for lb in labels:
                        threading.Thread(target=send_telegram_ndarray, args=(vis, lb), daemon=True).start()

            if not detected and last_detect_time is not None:
                if time.time() - last_detect_time > DEBOUNCE_SEC:
                    gpio_all(0)

            time.sleep(0.005)

    finally:
        gpio_all(0)
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Đã tắt GPIO & giải phóng tài nguyên.")

if __name__ == "__main__":
    main()

#pip install --upgrade pip setuptools wheel
#pip install torch torchvision torchaudio
#pip install ultralytics
#pip install opencv-python
#pip install requests numpy

