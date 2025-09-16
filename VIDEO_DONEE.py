from ultralytics import YOLO
import cv2
import requests

model_path = r"D:\fire1.pt"
video_path = r"D:\BREAKING_ Massive house fire causes collapse in Chicago, officials say.mp4"  # ← غيره لو عايز تشغّل كاميرا: video_path = 0


bot_token = "7439847095:AAG_4Fv2vCWUDZLrPJRMqZHh_gmVzaoXQ2M"
chat_id = "6351430058"

def send_fire_alert_telegram(photo_path):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    data = {"chat_id": chat_id, "caption": "🔥   DETECTED FIREEE !!! ."}
    files = {"photo": open(photo_path, "rb")}
    response = requests.post(url, data=data, files=files)

    if response.status_code == 200:
        print("Notification Sent Successfully")
    else:
        print(" Failed To Send Notification", response.text)

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

fire_detected_once = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    fire_detected = False

    for result in results:
        boxes = result.boxes
        if boxes is not None and boxes.xyxy is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                if class_name.lower() == 'fire':
                    fire_detected = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0]) * 100

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{class_name} {conf:.1f}%"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if fire_detected and not fire_detected_once:
        fire_detected_once = True
        alert_image_path = "detected_fire.jpg"
        cv2.imwrite(alert_image_path, frame)
        send_fire_alert_telegram(alert_image_path)

    cv2.imshow("Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
