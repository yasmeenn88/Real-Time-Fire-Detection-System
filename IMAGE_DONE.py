from ultralytics import YOLO
import cv2
import requests


model_path = r"D:\fire1.pt"
image_path = r"D:\نار.jpg"


bot_token = "7439847095:AAG_4Fv2vCWUDZLrPJRMqZHh_gmVzaoXQ2M"
chat_id = "6351430058"


def send_fire_alert_telegram(photo_path):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    data = {"chat_id": chat_id, "caption": "🔥 FIRE DETECTED NOW!"}
    files = {"photo": open(photo_path, "rb")}
    response = requests.post(url, data=data, files=files)

    if response.status_code == 200:
        print("Notification Sent Successfully")
    else:
        print("Failed To Send Notification", response.text)


model = YOLO(model_path)
results = model(image_path)


image = cv2.imread(image_path)
fire_detected = False

for result in results:
    for box in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        class_id = int(class_id)
        class_name = model.names[class_id]

        if class_name.lower() == 'fire':
            fire_detected = True

            # إحداثيات + رسم
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{class_name.upper()} {score * 100:.1f}%"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

if fire_detected:
    print("🔥 FIRE DETECTED!")
    cv2.imwrite(image_path, image)
    send_fire_alert_telegram(image_path)
else:
    print(" No fire detected.")
