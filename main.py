import cv2
import datetime
import threading
import requests
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")  # đổi sang file model của bạn

# Telegram config
BOT_TOKEN = "8169620330:AAEO_qBS1vXnA4Xb3jM4m2EOVQeWylhfzFU"
CHAT_ID = "-1002918477000"
TELEGRAM_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

last_alert = None


def alert(img, label):
    global last_alert

    if (last_alert is None) or (
        (datetime.datetime.utcnow() - last_alert).total_seconds() > 5
    ):
        last_alert = datetime.datetime.utcnow()
        filename = "alert.jpg"
        cv2.imwrite(filename, cv2.resize(img, dsize=None, fx=0.5, fy=0.5))

        thread = threading.Thread(target=send_telegram, args=(filename, label))
        thread.start()


def send_telegram(filename, label):
    try:
        with open(filename, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": CHAT_ID, "caption": f"🚨 {label} 🚨"}
            requests.post(TELEGRAM_URL, data=data, files=files)
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read video")
        break

    frame_resized = cv2.resize(frame, (800, 640))
    results = model(frame_resized)[0]

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
            frame_resized = results.plot()  # giữ bounding box
            for lb in labels:
                alert(frame_resized, lb)

    cv2.imshow("Stream", frame_resized)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
