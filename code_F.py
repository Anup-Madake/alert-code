import json
import time
from collections import deque, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import curve_fit
from ultralytics import YOLO


# ==============================
# Config (paths updated to your setup)
# ==============================
DEFAULT_MODEL = r"C:\Users\Mech.coep\OneDrive\Desktop\Lightray\Anup_try\yolov8_aug.pt"
DEFAULT_DISTANCE_JSON = r"C:\Users\Mech.coep\OneDrive\Desktop\Lightray\Anup_try\Latest_distance.json"

AREA_SWITCH = 113_520     # choose log vs power model by bbox area
EMA_ALPHA_V = 0.35        # smoothing for velocity
EMA_ALPHA_D = 0.20        # smoothing for distance (area->distance noise)
CENTER_ZONE_FRAC = 0.30   # center zone width fraction (e.g., middle 30% of frame)

# Danger logic (tune to taste)
TTC_LIMIT_SEC = 2.5
VEL_MIN_ALERT = 3.0       # m/s
DANGER_PROX_DIST = 3.0    # m
VEL_SLOW_MAX = 3.0        # m/s
VEL_SLOW_MIN = 1.0        # m/s


# ==============================
# Functions: distance models
# ==============================
def log_func(x, a, b):
    return a * np.log(x) + b

def power_func(x, a, b):
    return a * np.power(x, b)

class DistanceModelBank:
    """Cache per-class curve-fit params; load JSON once."""
    def __init__(self, json_path: str):
        self.json_path = json_path
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.cache = {}

    def get_params(self, class_name: str):
        key = class_name.lower()
        if key in self.cache:
            return self.cache[key]

        if key not in self.data:
            self.cache[key] = None
            return None

        vals = self.data[key]
        x1 = np.asarray(vals["x_values1"], dtype=float)
        y1 = np.asarray(vals["y_values1"], dtype=float)
        x2 = np.asarray(vals["x_values2"], dtype=float)
        y2 = np.asarray(vals["y_values2"], dtype=float)

        try:
            log_params, _ = curve_fit(
                log_func, np.clip(x1, 1e-6, None), y1, p0=(1.0, 0.0), maxfev=5000
            )
        except Exception:
            log_params = (1.0, 0.0)

        try:
            power_params, _ = curve_fit(
                power_func, np.clip(x2, 1e-6, None), y2, p0=(1.0, -0.5), maxfev=5000
            )
        except Exception:
            power_params = (1.0, -0.5)

        self.cache[key] = (tuple(log_params), tuple(power_params))
        return self.cache[key]

    def estimate_distance(self, class_name: str, area: float) -> float | None:
        params = self.get_params(class_name)
        if params is None:
            return None
        log_params, power_params = params
        a = max(float(area), 1e-6)
        try:
            if a > AREA_SWITCH:
                d = float(log_func(a, *log_params))
            else:
                d = float(power_func(a, *power_params))
            if not np.isfinite(d) or d < 0:
                return None
            return d
        except Exception:
            return None


# ==============================
# Helper math
# ==============================
def ema(prev: float | None, new: float, alpha: float) -> float:
    if prev is None:
        return new
    return (1 - alpha) * prev + alpha * new

def central_difference(seq_vals: deque, seq_ts: deque) -> float:
    """Central difference acceleration using last 3 points; returns 0 if insufficient."""
    if len(seq_vals) < 3 or len(seq_ts) < 3:
        return 0.0
    v_m1, v_0, v_p1 = seq_vals[-3], seq_vals[-2], seq_vals[-1]
    t_m1, t_0, t_p1 = seq_ts[-3], seq_ts[-2], seq_ts[-1]
    dt = (t_p1 - t_m1)
    if dt <= 0:
        return 0.0
    return (v_p1 - v_m1) / dt

def ttc_seconds(distance_m: float, velocity_mps: float) -> float:
    if not np.isfinite(distance_m) or distance_m <= 0:
        return float("inf")
    v = max(velocity_mps, 1e-6)
    return distance_m / v

def is_center_zone(x_center: float, frame_w: int) -> bool:
    left = frame_w * (0.5 - CENTER_ZONE_FRAC / 2)
    right = frame_w * (0.5 + CENTER_ZONE_FRAC / 2)
    return left <= x_center <= right

def motion_direction(prev_c: tuple[float, float] | None, cur_c: tuple[float, float]) -> str:
    if prev_c is None:
        return "Center"
    dx = cur_c[0] - prev_c[0]
    dy = cur_c[1] - prev_c[1]
    if abs(dx) > abs(dy):
        return "Right" if dx > 0 else "Left"
    return "Down" if dy > 0 else "Up"


# ==============================
# Visual helpers
# ==============================
def draw_panel(frame, x_left: int, y_top: int, lines: list[str], color=(255, 255, 255)):
    """Draw a semi-transparent info panel with text lines."""
    pad = 4
    line_h = 16
    width = max([cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for s in lines] + [120]) + 2 * pad
    height = line_h * len(lines) + 2 * pad
    y_top = max(0, y_top)
    x_left = max(0, x_left)
    y_bot = min(frame.shape[0], y_top + height)
    x_right = min(frame.shape[1], x_left + width)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_left, y_top), (x_right, y_bot), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    y = y_top + pad + 12
    for s in lines:
        cv2.putText(frame, s, (x_left + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += line_h


# ==============================
# Tracking state per ID
# ==============================
class TrackState:
    def __init__(self):
        self.prev_center = None
        self.dist_ema = None
        self.v_ema = None
        self.t_hist = deque(maxlen=64)   # timestamps for distance history
        self.d_hist = deque(maxlen=64)   # smoothed distance history
        self.vel_hist = deque(maxlen=64) # smoothed velocity history
        self.vel_ts = deque(maxlen=64)   # timestamps for velocity history


# ==============================
# Main processing
# ==============================
def process_video(
    input_path: str,
    output_path: str,
    model_path: str = DEFAULT_MODEL,
    distance_json: str = DEFAULT_DISTANCE_JSON,
    device: str = "cuda",
    imgsz: int = 640,
    half: bool = True,
    display: bool = True,
):
    # Load model
    model = YOLO(model_path)
    if device:
        try:
            model.to(device)
        except Exception:
            pass

    # Video IO
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: could not open input video: {input_path}")
        return

    # Read first frame to set size
    ok, frame = cap.read()
    if not ok:
        print("Error: could not read first frame.")
        cap.release()
        return

    h, w = frame.shape[:2]
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

    # Distance models
    dmb = DistanceModelBank(distance_json)

    # Per-ID states
    states: dict[int, TrackState] = defaultdict(TrackState)

    # Loop
    frame_idx = 0
    while True:
        if frame_idx > 0:
            ok, frame = cap.read()
            if not ok:
                break
        frame_idx += 1

        # Timestamp from the video stream (seconds)
        t_now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not np.isfinite(t_now):
            t_now = time.time()

        # Run tracker
        results = model.track(
            frame,
            persist=True,
            imgsz=imgsz,
            half=half if device == "cuda" else False,
            verbose=False,
        )

        if results and hasattr(results[0], "boxes") and results[0].boxes:
            boxes = results[0].boxes
            has_ids = hasattr(boxes, "id") and boxes.id is not None
            has_xywh = hasattr(boxes, "xywh")
            if has_ids and has_xywh:
                track_ids = boxes.id.int().cpu().tolist()
                boxes_xywh = boxes.xywh.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                classes_idx = boxes.cls.int().cpu().tolist()
                names_map = results[0].names

                for (x, y, bw, bh), tid, conf, cls_i in zip(boxes_xywh, track_ids, confidences, classes_idx):
                    cname = names_map.get(cls_i, str(cls_i))
                    area = float(bw * bh)
                    cx, cy = float(x), float(y)

                    # Distance estimate (smoothed)
                    d_raw = dmb.estimate_distance(cname, area)
                    state = states[tid]
                    if d_raw is not None:
                        d_smooth = ema(state.dist_ema, d_raw, EMA_ALPHA_D)
                        state.dist_ema = d_smooth
                    else:
                        d_smooth = state.dist_ema  # may be None

                    # Update distance history
                    state.d_hist.append(d_smooth if d_smooth is not None else np.nan)
                    state.t_hist.append(t_now)

                    # Compute instantaneous velocity as robust slope over small window (~0.7 s)
                    v_inst = 0.0
                    window_sec = 0.7
                    times = np.array(state.t_hist, dtype=float)
                    dists = np.array(state.d_hist, dtype=float)
                    good = np.isfinite(dists)
                    times = times[good]
                    dists = dists[good]
                    if len(times) >= 2:
                        t_end = times[-1]
                        mask = times >= (t_end - window_sec)
                        tw, dw = times[mask], dists[mask]
                        if len(tw) >= 2:
                            A = np.vstack([tw - tw[0], np.ones_like(tw)]).T
                            try:
                                a, _b = np.linalg.lstsq(A, dw, rcond=None)[0]
                                v_inst = abs(a)  # |slope| in m/s
                            except Exception:
                                v_inst = 0.0

                    # EMA smooth velocity
                    state.v_ema = ema(state.v_ema, v_inst, EMA_ALPHA_V)
                    v_sm = state.v_ema if state.v_ema is not None else v_inst

                    # Acceleration via central difference on velocity history
                    state.vel_hist.append(v_sm)
                    state.vel_ts.append(t_now)
                    a_cd = central_difference(state.vel_hist, state.vel_ts)

                    # Direction from center motion
                    direction = motion_direction(state.prev_center, (cx, cy))
                    state.prev_center = (cx, cy)

                    # Danger logic
                    ttc = float("inf")
                    if (d_smooth is not None) and (v_sm is not None):
                        ttc = ttc_seconds(d_smooth, v_sm)

                    # approaching heuristic (distance decreasing)
                    approaching = False
                    if len(dists) >= 2:
                        approaching = dists[-1] < dists[-2]

                    # center-zone gating
                    in_center = is_center_zone(cx, w)

                    red = False
                    if in_center and approaching:
                        if (ttc < TTC_LIMIT_SEC and v_sm > VEL_MIN_ALERT):
                            red = True
                        elif (d_smooth is not None) and (d_smooth < DANGER_PROX_DIST) and (VEL_SLOW_MIN <= v_sm <= VEL_SLOW_MAX):
                            red = True

                    box_color = (0, 0, 255) if red else (0, 255, 0)

                    # Draw box
                    x1 = int(cx - bw / 2)
                    y1 = int(cy - bh / 2)
                    x2 = int(cx + bw / 2)
                    y2 = int(cy + bh / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Info panel
                    info_lines = [
                        f"ID: {tid}  {cname}  {conf:.2f}",
                        f"Dist: {0.0 if d_smooth is None else d_smooth:.2f} m",
                        f"Vel:  {v_sm if v_sm is not None else 0.0:.2f} m/s",
                        f"Acc:  {a_cd:.2f} m/s^2",
                        f"TTC:  {ttc if np.isfinite(ttc) else 999.0:.2f} s",
                        f"Dir:  {direction}",
                        f"Center: {'Y' if in_center else 'N'}  Appr: {'Y' if approaching else 'N'}",
                    ]
                    panel_y = max(0, y1 - 110)
                    draw_panel(frame, x1, panel_y, info_lines, color=(255, 255, 255) if not red else (0, 0, 255))

        # Center zone guide (optional)
        cz_w = int(w * CENTER_ZONE_FRAC)
        cz_x1 = (w - cz_w) // 2
        cz_x2 = cz_x1 + cz_w
        cv2.rectangle(frame, (cz_x1, 0), (cz_x2, h), (255, 255, 255), 1)

        # Write & display
        out.write(frame)
        if display:
            cv2.imshow("YOLOv8 Tracking (Video)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# ==============================
# Hardcoded run (paths updated to your setup)
# ==============================
if __name__ == "__main__":
    input_video_path  = r"C:\Users\Mech.coep\OneDrive\Desktop\Lightray\Anup_try\Test_Video17.mp4"
    output_video_path = r"C:\Users\Mech.coep\OneDrive\Desktop\Lightray\Anup_try\output_video.avi"

    process_video(
        input_path=input_video_path,
        output_path=output_video_path,
        model_path=DEFAULT_MODEL,
        distance_json=DEFAULT_DISTANCE_JSON,
        device="cuda",   # change to "cpu" if no CUDA GPU
        imgsz=640,
        half=True,       # set False if device="cpu"
        display=True,
    )
