import tkinter as tk
import threading
import onnxruntime
import cv2
import numpy as np
from PIL import Image, ImageTk

class DetectionAlarmUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detection Alarm")

        # Application Title
        self.title_label = tk.Label(root, text="Detection Alarm", font=("Arial", 20))
        self.title_label.pack(pady=20)

        # Confirmation Button
        self.confirm_button = tk.Button(root, text="확인", command=self.start_inference, font=("Arial", 14))
        self.confirm_button.pack(pady=20)

        # Termination Button
        self.terminate_button = tk.Button(root, text="종료", command=self.stop_inference, font=("Arial", 14))
        self.terminate_button.pack(pady=10)

        # Person confidence display
        self.confidence_label = tk.Label(root, text="Detection Confidence: 0.0", font=("Arial", 14))
        self.confidence_label.pack(padx=10)

        # Alarm sound status display
        self.alarm_status = tk.StringVar()
        self.alarm_status.set("Alarm Sound: OFF")
        self.alarm_status_label = tk.Label(root, textvariable=self.alarm_status, font=("Arial", 14))
        self.alarm_status_label.pack(pady=20)

        # Initialize the camera feed
        self.box = None
        self.inference_thread = None
        self.stop_flag = threading.Event()
        self.init_camera()

    def init_camera(self):
        """Initialize the camera and create a canvas for video feed."""
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.canvas = tk.Canvas(self.root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack(padx=10, pady=10)
        self.update_camera_feed()

    def draw_boxes_on_frame(self, frame, box):
            # Convert [cx, cy, w, h] format to [x1, y1, x2, y2]
        cx, cy, w, h = box
        #print(cx, cy, w, h)
        x1 = int((cx - w / 2))
        y1 = int((cy - h / 2))
        x2 = int((cx + w / 2))
        y2 = int((cy + h / 2))
        #print(x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle in green color

    def update_camera_feed(self):
        """Update the video frames in the UI."""
        ret, frame = self.vid.read()
        if self.stop_flag.is_set():
            self.box = None
        if ret:
            # Resize the frame to 640x640
            frame = cv2.resize(frame, (640, 640))
            # If boxes have been detected, draw them on the frame
            if self.box is not None:
                self.draw_boxes_on_frame(frame, self.box)
            else:
                pass
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.after(10, self.update_camera_feed)

    def preprocess_image(self, image, input_size):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size))
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image /= 255.0
        image = np.expand_dims(image, axis=0)
        return image
            
        
    def postprocess_output(self, output, number=0, threshold=0.9):
        predictions = output[0].squeeze(0)
        detected = False  # Variable to check if a person was detected
        box = None  # Initial value
        for pred in predictions:
            person_confidence = pred[5+number]
            if person_confidence > threshold:
                self.confidence_label.config(text=f"Confidence: {person_confidence:.2f}")
                self.alarm_status.set("Alarm Sound: ON")
                box = pred[0:4]
                detected = True
                break  # Exit the loop if a person was detected
        if not detected:
            self.confidence_label.config(text="No detected")
            self.alarm_status.set("Alarm Sound: OFF")
        return box

    def start_inference(self):
        # Reset the stop flag
        self.stop_flag.clear()

        # Launch YOLOv5 inference in a separate thread
        self.inference_thread = threading.Thread(target=self.run_yolov5_inference)
        self.inference_thread.start()

    def run_yolov5_inference(self):
        model_path = "yolov5s.onnx"
        session = onnxruntime.InferenceSession(model_path)

        while not self.stop_flag.is_set():
            ret, frame = self.vid.read()  # self.vid에서 직접 프레임 읽기
            if not ret:
                print("Error: Could not read frame.")
                break

            input_tensor = self.preprocess_image(frame, 640)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            output = session.run([output_name], {input_name: input_tensor})
            self.box = self.postprocess_output(output)

    def stop_inference(self):
        # Set the stop flag to terminate the YOLOv5 inference thread
        self.stop_flag.set()
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(2.0)  # Wait for up to 2 seconds
        self.confidence_label.config(text="Detection Confidence: 0.0")
        self.alarm_status.set("Alarm Sound: OFF")
        print("YOLOv5 inference stopped.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionAlarmUI(root)
    app.run()
