import tkinter as tk
from tkinter import simpledialog
from datetime import datetime

import cv2
import numpy as np
import onnxruntime
from PIL import Image

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
        person_confidence = pred[5] # 0: person, 43: knife 
        if person_confidence > threshold:
            print(f"detected with confidence: {person_confidence}")
            
def main():
    model_path = "yolov5s.onnx"
    session = onnxruntime.InferenceSession(model_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return
    
    while True:
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
