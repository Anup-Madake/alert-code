import cv2
import time

# === Crash Warning Logic ===
def compute_distance_to_collision(d_initial, v_rel, a_rel, u_rel):
    if a_rel == 0:
        return float('inf')
    return d_initial - (v_rel**2 - u_rel**2) / (2 * a_rel)

def compute_jerk(a_rel, u_rel):
    if u_rel == 0:
        return float('inf')
    return abs((a_rel**2) / u_rel)

def compute_safe_stopping_distance(u_rel, a_rel):
    if a_rel == 0:
        return float('inf')
    return u_rel * 2.5 + (u_rel**2) / (2 * a_rel)

def check_crash_warning(v_rel_list, a_rel_list, d_initial,
                        Distance_Threshold=8, Jerk_threshold=8, Braking_distance_threshold=2):
    if not v_rel_list or not a_rel_list:
        return "Insufficient data"

    u_rel = v_rel_list[0]
    v_rel = v_rel_list[-1]
    a_rel = a_rel_list[-1]

    if v_rel >= 0 and a_rel >= 0:
        d1 = compute_distance_to_collision(d_initial, v_rel, a_rel, u_rel)

        if d1 < Distance_Threshold:
            jerk = compute_jerk(a_rel, v_rel)
            if jerk < Jerk_threshold:
                stopping_distance = compute_safe_stopping_distance(v_rel, a_rel)
                if stopping_distance < Braking_distance_threshold:
                    return "⚠️ Crash warning!"
                else:
                    return "✅ Safe: Enough braking distance"
            else:
                return "✅ Safe: High braking responsiveness"
        else:
            return "✅ Safe: Enough distance to stop"
    else:
        return "✅ Safe: No relative threat"

# === Video + Visual Logic ===
cap = cv2.VideoCapture("C:/Users/Mech.coep/OneDrive/Desktop/try/Test_Video2q.mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
focal_length = 700  # Adjust based on calibration
real_height = 1.5   # Average vehicle height in meters

# Tracking data
v_rel_list, a_rel_list, d_list = [], [], []
prev_distance = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    object_box = (width//2 - 50, height//2 - 50, 100, 100)  # Dummy box

    # Distance estimation using object height
    pixel_height = object_box[3]
    if pixel_height == 0:
        continue
    distance = (real_height * focal_length) / pixel_height
    d_list.append(distance)

    # Velocity estimation
    if prev_distance is not None:
        v_rel = (prev_distance - distance) * fps
        v_rel_list.append(v_rel)

        # Acceleration estimation
        if len(v_rel_list) >= 2:
            a_rel = (v_rel_list[-1] - v_rel_list[-2]) * fps
            a_rel_list.append(a_rel)

            # Check crash status
            d_initial = d_list[-2] if len(d_list) >= 2 else distance
            result = check_crash_warning(v_rel_list, a_rel_list, d_initial)

            # Annotate result
            cv2.putText(frame, result, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if "⚠️" in result else (0, 255, 0), 2)

    prev_distance = distance

    # Draw bounding box
    x, y, w, h = object_box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Crash Detection", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
