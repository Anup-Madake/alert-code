import cv2
import time
import numpy as np
from collections import deque
from ultralytics import YOLO

# === TTC and Impact Logic ===
def compute_ttc(u, a, d):
    if a > 0:
        delta = u**2 + 2 * a * d
        if delta < 0:
            return None, None, "No Collision"
        ttc = (-u + np.sqrt(delta)) / a
        v_impact = u + a * ttc
        return ttc, v_impact, "Crash Likely"
    elif a == 0:
        if u <= 0:
            return None, None, "No Collision"
        ttc = d / u
        return ttc, u, "Crash Likely"
    else:
        A = -a
        delta = u**2 - 2 * A * d
        if delta < 0:
            return None, None, "No Collision"
        ttc = (-u + np.sqrt(delta)) / a
        v_impact = u + a * ttc
        d_stop = u**2 / (2 * A)
        if d <= d_stop:
            return ttc, v_impact, "Crash Likely"
        else:
            return None, None, "No Collision"

def classify_impact(v_impact, threshold=2.0):
    if v_impact is None:
        return "No Impact"
    elif abs(v_impact) < threshold:
        return "Minor Impact"
    else:
        return "Major Impact"

# === TTC Video Analyzer ===
class TTCVideoAnalyzer:
    def __init__(self, video_path, model_path='yolov8n.pt'):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.model = YOLO(model_path)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.focal_length = 700
        self.real_height = 1.5  # in meters

        # === Setup video writer ===
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = r"C:\Users\Mech.coep\Desktop\Crash_Output.mp4"
        self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        # Motion tracking
        self.prev_distance = None
        self.velocity_history = deque(maxlen=6)
        self.height_smoothing = deque(maxlen=5)

    def run(self):
        prev_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            annotated = self.process_frame(frame)

            # Show FPS
            new_time = time.time()
            fps = 1 / max(new_time - prev_time, 1e-5)
            prev_time = new_time
            cv2.putText(annotated, f"FPS: {fps:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Accurate TTC Crash Detection", annotated)
            self.writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Output video saved at: C:\\Users\\Mech.coep\\Desktop\\Crash_Output.mp4")

    def process_frame(self, frame):
        results = self.model.track(frame, persist=True)
        annotated = frame.copy()

        if results and results[0].boxes:
            boxes = results[0].boxes
            classes = boxes.cls.cpu().numpy()
            heights = []
            bboxes = []

            for box, cls in zip(boxes.xyxy, classes):
                if int(cls) in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    h = max(y2 - y1, 1)
                    heights.append(h)
                    bboxes.append((x1, y1, x2, y2))

            if bboxes:
                max_idx = np.argmax(heights)
                x1, y1, x2, y2 = bboxes[max_idx]
                h = heights[max_idx]
                self.height_smoothing.append(h)
                smoothed_h = np.mean(self.height_smoothing)

                # === Distance Estimation ===
                distance = (self.real_height * self.focal_length) / max(smoothed_h, 1)
                distance = round(distance, 2)

                # === Show Distance on Frame ===
                cv2.putText(annotated, f"Distance: {distance:.2f} m", (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

                # === Velocity + Acceleration ===
                if self.prev_distance is not None:
                    velocity = (self.prev_distance - distance) * self.fps
                    self.velocity_history.append(velocity)

                    if len(self.velocity_history) >= 2:
                        a = (self.velocity_history[-1] - self.velocity_history[-2]) * self.fps
                        u = self.velocity_history[-2]
                        d = distance

                        ttc, v_impact, status = compute_ttc(u, a, d)
                        impact = classify_impact(v_impact)

                        if d <= 3.0:
                            cv2.putText(annotated, "⚠️ Danger: Object very close!", (30, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
                        else:
                            cv2.putText(annotated, f"TTC: {ttc:.2f}s" if ttc else "TTC: -", (30, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

                        cv2.putText(annotated, f"Impact: {impact}", (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                        cv2.putText(annotated, f"Status: {status}", (30, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

                self.prev_distance = distance

                # === Draw Box and Center ===
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                box_color = (0, 0, 255) if distance <= 3.0 else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

        return annotated

# === Main Entry Point ===
def main():
    video_path = r"C:\Users\Mech.coep\OneDrive\Desktop\try\Test_Video11.mp4"
    analyzer = TTCVideoAnalyzer(video_path)
    analyzer.run()

if __name__ == "__main__":
    main()
