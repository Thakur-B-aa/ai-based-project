import pyttsx3
import time
engine = pyttsx3.init()
engine.say(" object detected")
engine.runAndWait()

from ultralytics import YOLO
# Load the AI model
model = YOLO("yolo11n.pt")
engine =pyttsx3.init()
# Start the laptop camera
results = model.predict(
    source=0,
    conf=0.5
)
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        object_name = model.names[class_id]
        print("Object detected:", object_name)
        engine.say(object_name)
        engine.runAndWait()

