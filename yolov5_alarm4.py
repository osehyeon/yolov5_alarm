import tkinter as tk
import threading
import onnxruntime
import cv2
import numpy as np
from PIL import Image, ImageTk

class DetectionAlarmUI:
    def __init__(self, root):
        self.root = root
        self.stop_flag = threading.Event() 
        self.box = None
        self.inference_thread = None
        self.setup_ui()
        self.init_camera()

    def setup_ui(self):
        """Setup the UI elements."""
        self.root.title("Detection Alarm")

        # Widgets setup
        self.add_widget(tk.Label(self.root, text="Detection Alarm", font=("Arial", 20)), pady=20)
        self.add_widget(tk.Button(self.root, text="확인", command=self.start_inference, font=("Arial", 14)), pady=20)
        self.add_widget(tk.Button(self.root, text="종료", command=self.stop_inference, font=("Arial", 14)), pady=10)
        self.confidence_label = self.add_widget(tk.Label(self.root, text="Detection Confidence: 0.0", font=("Arial", 14)), padx=10)
        self.alarm_status = tk.StringVar(value="Alarm Sound: OFF")
        self.add_widget(tk.Label(self.root, textvariable=self.alarm_status, font=("Arial", 14)), pady=20)

    def add_widget(self, widget, **pack_args):
        """Utility to add and pack a widget in one step."""
        widget.pack(**pack_args)
        return widget

    def init_camera(self):
        """Initialize the camera."""
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.canvas = tk.Canvas(self.root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack(padx=10, pady=10)
        self.update_camera_feed()

    def update_camera_feed(self):
        """Update the video frames in the UI."""
        frame = self.get_frame()
        if frame is not None:
            self.show_frame(frame)
        self.root.after(10, self.update_camera_feed)

    def get_frame(self):
        """Retrieve a frame from the video source."""
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.resize(frame, (640, 640))
            if not self.stop_flag.is_set() and self.box is not None:
                self.draw_boxes_on_frame(frame, self.box)
            return frame
        return None

    def show_frame(self, frame):
        """Display a frame on the canvas."""
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def draw_boxes_on_frame(self, frame, box):
        cx, cy, w, h = box
        x1, y1 = int((cx - w / 2)), int((cy - h / 2))
        x2, y2 = int((cx + w / 2)), int((cy + h / 2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def preprocess_image(self, image, input_size):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size))
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image /= 255.0
        return np.expand_dims(image, axis=0)
            
    def postprocess_output(self, output, number=0, threshold=0.9):
        predictions = output[0].squeeze(0)
        for pred in predictions:
            person_confidence = pred[5+number]
            if person_confidence > threshold:
                self.confidence_label.config(text=f"Confidence: {person_confidence:.2f}")
                self.alarm_status.set("Alarm Sound: ON")
                return pred[0:4]
        return None

    def start_inference(self):
        """Start the inference thread."""
        self.stop_flag.clear()
        self.inference_thread = threading.Thread(target=self.run_yolov5_inference)
        self.inference_thread.start()

    def run_yolov5_inference(self):
        """Run inference on YOLOv5 model."""
        model_path = "yolov5s.onnx"
        session = onnxruntime.InferenceSession(model_path)
        input_name, output_name = session.get_inputs()[0].name, session.get_outputs()[0].name

        while not self.stop_flag.is_set():
            frame = self.get_frame()
            if frame is None:
                continue
            input_tensor = self.preprocess_image(frame, 640)
            output = session.run([output_name], {input_name: input_tensor})
            self.box = self.postprocess_output(output)

    def stop_inference(self):
        """Stop the inference thread."""
        self.stop_flag.set()
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(2.0)
        self.confidence_label.config(text="Detection Confidence: 0.0")
        self.alarm_status.set("Alarm Sound: OFF")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionAlarmUI(root)
    app.run()
