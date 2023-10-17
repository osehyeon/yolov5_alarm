import tkinter as tk
from tkinter import simpledialog
from datetime import datetime

import cv2
import numpy as np
import onnxruntime
from PIL import Image

def get_time_interval():
    root = tk.Tk()
    root.withdraw()  # hide the main window

    start_time = simpledialog.askstring("Input", "Start Time (HH:MM format):", parent=root)
    end_time = simpledialog.askstring("Input", "End Time (HH:MM format):", parent=root)

    return start_time, end_time

def is_within_time_interval(start_time, end_time):
    current_time = datetime.now().time()
    start_hour, start_minute = map(int, start_time.split(":"))
    end_hour, end_minute = map(int, end_time.split(":"))

    start_time_obj = datetime.strptime(start_time, '%H:%M').time()
    end_time_obj = datetime.strptime(end_time, '%H:%M').time()

    return start_time_obj <= current_time <= end_time_obj

def preprocess_image(image, input_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size))
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_output(output, threshold=0.9):
    predictions = output[0].squeeze(0)
    for pred in predictions:
        person_confidence = pred[5]
        if person_confidence > threshold:
            print(f"Person detected with confidence: {person_confidence}")
            
def main():
    model_path = "yolov5s.onnx"
    session = onnxruntime.InferenceSession(model_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    # UI를 사용하여 사용자로부터 시간 간격을 받아옵니다.
    start_time, end_time = get_time_interval()

    while True:
        # 현재 시간이 사용자가 지정한 시간 범위 내에 있는지 확인합니다.
        if not is_within_time_interval(start_time, end_time):
            continue

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        input_tensor = preprocess_image(frame, 640)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        output = session.run([output_name], {input_name: input_tensor})
        postprocess_output(output)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
