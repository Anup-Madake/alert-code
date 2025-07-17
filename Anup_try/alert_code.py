import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import curve_fit
from ultralytics import YOLO
import pygame
import threading
import os
from queue import Queue

# Initialize pygame mixer
pygame.mixer.init()

def log_func(x, a, b): return a * np.log(x) + b

def power_func(x, a, b): return a * np.power(x, b)

def compute_ttc(u, a, d):
    if a > 0:
        delta = u**2 + 2*a*d
        if delta < 0: return None, None, "No Collision"
        ttc = (-u + np.sqrt(delta)) / a
        return ttc, u + a*ttc, "Crash Likely"
    elif a == 0:
        return (d / u, u, "Crash Likely") if u > 0 else (None, None, "No Collision")
    else:
        A = -a
        delta = u**2 - 2*A*d
        if delta < 0: return None, None, "No Collision"
        ttc = (-u + np.sqrt(delta)) / a
        d_stop = u**2 / (2*A)
        return (ttc, u + a*ttc, "Crash Likely") if d <= d_stop else (None, None, "No Collision")

def classify_impact(v_impact, threshold=2.0):
    if v_impact is None: return "No Impact"
    return "Minor Impact" if abs(v_impact) < threshold else "Major Impact"

def calculate_distance_models(class_name):
    with open("distance_new.json", 'r') as file:
        data = json.load(file)
    if class_name not in data:
        raise KeyError(f"{class_name} not found in JSON.")
    x1, y1 = np.array(data[class_name]['x_values1']), np.array(data[class_name]['y_values1'])
    x2, y2 = np.array(data[class_name]['x_values2']), np.array(data[class_name]['y_values2'])
    return curve_fit(log_func, x1, y1)[0], curve_fit(power_func, x2, y2)[0]

def plot_velocity_acceleration(velocity_history, acceleration_history):
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    for tid, v in velocity_history.items():
        plt.plot(v, label=f"Vehicle {tid}")
    plt.title("Velocity of Vehicles")
    plt.xlabel("Frame")
    plt.ylabel("Velocity (m/s)")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    for tid, a in acceleration_history.items():
        plt.plot(a, label=f"Vehicle {tid}")
    plt.title("Acceleration of Vehicles")
    plt.xlabel("Frame")
    plt.ylabel("Acceleration (m/s²)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def play_alert(audio_path, direction=""):
    if os.path.exists(audio_path):
        try:
            print(f"🔊 Playing: {audio_path} for {direction}")
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"⚠️ Failed to play sound: {e}")
    else:
        print(f"❌ Missing alert sound: {audio_path}")

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n.pt")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter("Crash_Output_Final.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    allowed_classes = {'car', 'truck', 'bus', 'motorcycle', 'bike', 'bicycle', 'auto'}
    track_history, velocity_history, acceleration_history, prev_times = {}, {}, {}, {}
    height_smoothing = deque(maxlen=5)
    prev_frame_time = time.time()

    left_audio = "left_alert.mp3"
    right_audio = "right_alert.mp3"
    close_audio = "close_alert.mp3"

    alert_last_played = {}
    cooldown = 3  # seconds
    frame_center_x = width // 2
    center_tolerance = width // 10
    bottom_y_threshold = height * 0.75
    close_distance_threshold = 4.0
    min_area_to_consider = 3000

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, iou=0.5)
        if not results or not results[0].boxes:
            continue

        boxes = results[0].boxes
        if boxes.id is None:
            continue

        ids = boxes.id.int().cpu().tolist()
        coords = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = [results[0].names[i] for i in boxes.cls.int().cpu().tolist()]

        for box, tid, conf, cls in zip(coords, ids, confs, classes):
            if cls not in allowed_classes:
                continue

            x, y, w, h = box
            area = w * h

            if tid not in track_history:
                track_history[tid] = []
                velocity_history[tid] = []
                acceleration_history[tid] = []
                prev_times[tid] = []

            track_history[tid].append((x, y, w, h, area))
            prev_times[tid].append(time.time())

            if len(track_history[tid]) < 15:
                continue

            d_prev = track_history[tid][-15][4]
            d_now = area

            try:
                log_params, power_params = calculate_distance_models(cls)
                distance = log_func(area, *log_params) if area > 113520 else power_func(area, *power_params)
            except:
                height_smoothing.append(h)
                smoothed_h = np.mean(height_smoothing)
                distance = 1.5 * 700 / smoothed_h if smoothed_h > 0 else 5.0

            distance = round(distance, 2)
            t_elapsed = prev_times[tid][-1] - prev_times[tid][-15]
            velocity = (d_prev - d_now) / t_elapsed if t_elapsed > 0 else 0
            velocity_history[tid].append(velocity)

            if len(velocity_history[tid]) >= 2:
                v_prev = velocity_history[tid][-2]
                accel = (velocity - v_prev) / t_elapsed if t_elapsed > 0 else 0
                acceleration_history[tid].append(accel)
                ttc, v_impact, crash_status = compute_ttc(v_prev, accel, distance)
                impact_type = classify_impact(v_impact)
            else:
                accel, ttc, impact_type, crash_status = 0, None, "No Impact", "Tracking..."

            now = time.time()
            approaching = velocity > 0.05 or accel > 0.02
            overtaking = False
            if len(track_history[tid]) >= 2:
                x_prev = track_history[tid][-2][0]
                overtaking = abs(x - x_prev) > 1.5
            in_center = frame_center_x - center_tolerance <= x <= frame_center_x + center_tolerance
            is_close = distance <= close_distance_threshold and area > min_area_to_consider
            in_bottom = y > bottom_y_threshold

            if approaching and in_center and in_bottom and is_close and overtaking:
                last_play = alert_last_played.get(tid, 0)
                if now - last_play >= cooldown:
                    direction = "Left" if x < x_prev else "Right"
                    alert_path = left_audio if direction == "Left" else right_audio
                    print(f"ALERT TRIGGERED: tid={tid}, dist={distance:.2f}m, vel={velocity:.2f}, accel={accel:.2f}, area={area:.2f}")
                    play_alert(alert_path, direction)
                    alert_last_played[tid] = now
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
                alert_last_played.pop(tid, None)

            y_offset = int(y - h / 2 - 100)
            for text in [
                f"Track ID: {tid}",
                f"Conf: {conf:.2f}",
                f"Class: {cls}",
                f"Distance: {distance:.2f}m",
                f"Velocity: {velocity:.2f} m/s",
                f"Acceleration: {accel:.2f} m/s²",
                f"TTC: {ttc:.2f}s" if ttc else "TTC: -",
                f"Impact: {impact_type}",
                f"Status: {crash_status}"
            ]:
                cv2.putText(frame, text, (int(x - w/2), y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 15

            cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), color, 2)
            cv2.putText(frame, f"{cls} ({conf:.2f})", (int(x - w/2), int(y - h/2 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        new_frame_time = time.time()
        fps_text = 1 / (new_frame_time - prev_frame_time + 1e-5)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {fps_text:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out.write(frame)
        cv2.imshow("TTC + Tracking Filtered", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    plot_velocity_acceleration(velocity_history, acceleration_history)

if __name__ == "__main__":
    analyze_video("Test_Video17.mp4")