import tkinter as tk
from datetime import datetime
import threading
import onnxruntime
import cv2
import numpy as np

class MiracleMorningUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Miracle Morning")

        # Application Title
        self.title_label = tk.Label(root, text="Miracle Morning", font=("Arial", 20))
        self.title_label.pack(pady=20)

        # Display Current Time
        self.current_time_label = tk.Label(root, text="", font=("Arial", 24))
        self.current_time_label.pack(pady=20)
        self.update_time()

        time_frame = tk.Frame(self.root)
        time_frame.pack(pady=10)

        # Alarm Start Time
        self.start_hour_var = tk.StringVar(root)
        self.start_hour_var.set("오전")
        self.start_hour_dropdown = tk.OptionMenu(root, self.start_hour_var, "오전", "오후")
        self.start_hour_dropdown.pack(side=tk.LEFT, padx=5, in_=time_frame)

        self.start_time_var = tk.StringVar(root)
        self.start_time_entry = tk.Entry(root, textvariable=self.start_time_var, width=5, font=("Arial", 14))
        self.start_time_entry.pack(side=tk.LEFT, padx=5, in_=time_frame)
        self.start_time_var.set("07:00")

        # Alarm End Time
        self.end_hour_var = tk.StringVar(root)
        self.end_hour_var.set("오전")
        self.end_hour_dropdown = tk.OptionMenu(root, self.end_hour_var, "오전", "오후")
        self.end_hour_dropdown.pack(side=tk.LEFT, padx=5, in_=time_frame)

        self.end_time_var = tk.StringVar(root)
        self.end_time_entry = tk.Entry(root, textvariable=self.end_time_var, width=5, font=("Arial", 14))
        self.end_time_entry.pack(side=tk.LEFT, padx=5, in_=time_frame)
        self.end_time_var.set("08:00")

        # Confirmation Button
        self.confirm_button = tk.Button(root, text="확인", command=self.start_inference, font=("Arial", 14))
        self.confirm_button.pack(side=tk.LEFT, padx=5, in_=time_frame)

        # Termination Button
        self.terminate_button = tk.Button(root, text="종료", command=self.stop_inference, font=("Arial", 14))
        self.terminate_button.pack(side=tk.LEFT, padx=5, in_=time_frame)

        # Person confidence display
        self.confidence_label = tk.Label(root, text="Person Confidence: 0.0", font=("Arial", 14))
        self.confidence_label.pack(pady=10)

        # Alarm sound status display
        self.alarm_status = tk.StringVar()
        self.alarm_status.set("Alarm Sound: OFF")
        self.alarm_status_label = tk.Label(root, textvariable=self.alarm_status, font=("Arial", 14))
        self.alarm_status_label.pack(pady=10)

        self.inference_thread = None
        self.stop_flag = threading.Event()

    def is_within_time_interval(self, start_time, end_time):
        current_time = datetime.now().time()
        start_hour, start_minute = map(int, start_time.split(":"))
        end_hour, end_minute = map(int, end_time.split(":"))

        start_time_obj = datetime.strptime(start_time, '%H:%M').time()
        end_time_obj = datetime.strptime(end_time, '%H:%M').time()

        return start_time_obj <= current_time <= end_time_obj

    def update_time(self):
        current_time = datetime.now().strftime('%H:%M:%S')
        self.current_time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def preprocess_image(self, image, input_size):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size))
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image /= 255.0
        image = np.expand_dims(image, axis=0)
        return image
            
    def turn_off_alarm(self):
        self.alarm_status.set("Alarm Sound: OFF")
        
    def postprocess_output(self, output, threshold=0.9):
        predictions = output[0].squeeze(0)
        for idx, pred in enumerate(predictions):
            person_confidence = pred[5]
            if person_confidence > threshold:
                self.confidence_label.config(text=f"Person Confidence: {person_confidence:.2f}")
                self.alarm_status.set("Alarm Sound: ON")
                print(f"Person detected with confidence: {person_confidence}")
            elif person_confidence < threshold and idx == 0: 
                self.confidence_label.config(text=f"Person Confidence: 0.00")
                self.alarm_status.set("Alarm Sound: OFF")

    def start_inference(self):
        start_time = self.start_time_var.get()
        end_time = self.end_time_var.get()
        start_am_pm = self.start_hour_var.get()
        end_am_pm = self.end_hour_var.get()

        if start_am_pm == "오후" and not start_time.startswith("12"):
            start_time = str(int(start_time.split(':')[0]) + 12) + ":" + start_time.split(':')[1]
        if end_am_pm == "오후" and not end_time.startswith("12"):
            end_time = str(int(end_time.split(':')[0]) + 12) + ":" + end_time.split(':')[1]

        self.start_time = start_time  # 이 변수를 클래스 변수로 저장합니다.
        self.end_time = end_time      # 이 변수를 클래스 변수로 저장합니다.

        print(f"Alarm Start Time: {self.start_time}")
        print(f"Alarm End Time: {self.end_time}")
        
        # Reset the stop flag
        self.stop_flag.clear()

        # Launch YOLOv5 inference in a separate thread
        self.inference_thread = threading.Thread(target=self.run_yolov5_inference)
        self.inference_thread.start()

    def run_yolov5_inference(self):
        model_path = "yolov5s.onnx"
        session = onnxruntime.InferenceSession(model_path)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera not found.")
            return

        while not self.stop_flag.is_set():
            # 현재 시간이 사용자가 지정한 시간 범위 내에 있는지 확인합니다.
            if not self.is_within_time_interval(self.start_time, self.end_time):
                continue

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            input_tensor = self.preprocess_image(frame, 640)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            output = session.run([output_name], {input_name: input_tensor})
            self.postprocess_output(output)

        cap.release()
        cv2.destroyAllWindows()

    def stop_inference(self):
        # Set the stop flag to terminate the YOLOv5 inference thread
        self.stop_flag.set()
        self.inference_thread.join()  # Wait for the thread to finish

        print("YOLOv5 inference stopped.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = MiracleMorningUI(root)
    app.run()