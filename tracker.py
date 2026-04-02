import cv2
import time
import threading
import numpy as np
from collections import deque

# ============================================================
#  VELOCITY_TRACKR  —  fast-object edition
#  Trackers: CSRT (accurate) → KCF fallback → MOSSE fallback
#  Features: Kalman prediction, trajectory trail, multi-zone HUD
# ============================================================

# ── Kalman filter for predicting fast-moving objects ────────
class KalmanPredictor:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost       = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, cx, cy):
        m = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self.initialized:
            self.kf.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
            self.initialized = True
        self.kf.correct(m)
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

    def reset(self):
        self.initialized = False


# ── Threaded camera (no frame-drop lag) ─────────────────────
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # CAP_DSHOW = faster on Windows
        self.cap.set(cv2.CAP_PROP_FPS,          120)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)        # minimal buffer = less latency
        self.ret, self.frame = self.cap.read()
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.ret:
                return True, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        self.cap.release()


# ── Pick the best available tracker ─────────────────────────
def make_tracker():
    # CSRT = best accuracy for fast objects (slower but worth it)
    # KCF  = good balance
    # MOSSE= fastest but loses fast objects
    for name in ("CSRT", "KCF", "MOSSE"):
        try:
            return getattr(cv2, f"Tracker{name}_create")(), name
        except AttributeError:
            continue
    raise RuntimeError("No OpenCV tracker found. Install opencv-contrib-python.")


# ── Draw a filled rounded rectangle (for HUD panels) ────────
def draw_panel(frame, x, y, w, h, color, alpha=0.45):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)


# ── Corner bracket helper ────────────────────────────────────
def draw_corners(frame, x, y, wb, hb, color, size=20, thickness=3):
    for px, py, dx, dy in [
        (x,      y,      1,  1),
        (x+wb,   y,     -1,  1),
        (x,      y+hb,   1, -1),
        (x+wb,   y+hb,  -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx*size, py),       color, thickness)
        cv2.line(frame, (px, py), (px,           py+dy*size),color, thickness)


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
cam        = CameraStream(0)
time.sleep(0.6)

tracker, tracker_name = make_tracker()
kalman     = KalmanPredictor()
is_tracking= False
TRACK_SCALE= 0.6          # track at 60 % res — fast & accurate enough for CSRT

trail      = deque(maxlen=40)   # trajectory dots
velocities = deque(maxlen=10)   # for speed estimation
prev_center= None
prev_time  = time.time()
fps_display= 0
lost_frames= 0
MAX_LOST   = 8             # frames before declaring signal lost

# colours
CYAN   = (255, 220, 0)
GREEN  = (0, 255, 120)
RED    = (0, 60, 255)
ORANGE = (0, 165, 255)
GREY   = (70, 70, 70)
WHITE  = (220, 220, 220)
DARK   = (10, 10, 10)

print("╔══════════════════════════════╗")
print("║   VELOCITY_TRACKR  v9.0      ║")
print("╠══════════════════════════════╣")
print("║  [S] Select target           ║")
print("║  [R] Reset tracker           ║")
print("║  [T] Switch tracker type     ║")
print("║  [Q] Quit                    ║")
print("╚══════════════════════════════╝")
print(f"  Active tracker: {tracker_name}")

tracker_idx = 0
tracker_names_list = ["CSRT", "KCF", "MOSSE"]

while True:
    success, frame = cam.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    now         = time.time()
    dt          = max(now - prev_time, 1e-6)
    fps_display = 0.8 * fps_display + 0.2 * (1 / dt)   # smoothed FPS
    prev_time   = now

    # ── TRACKING LOGIC ───────────────────────────────────────
    if is_tracking:
        small        = cv2.resize(frame, (int(w*TRACK_SCALE), int(h*TRACK_SCALE)))
        ok, sm_bbox  = tracker.update(small)

        if ok:
            lost_frames = 0
            s  = 1 / TRACK_SCALE
            x  = int(sm_bbox[0] * s)
            y  = int(sm_bbox[1] * s)
            wb = int(sm_bbox[2] * s)
            hb = int(sm_bbox[3] * s)

            # clamp to frame
            x  = max(0, min(x,  w-1))
            y  = max(0, min(y,  h-1))
            wb = max(1, min(wb, w-x))
            hb = max(1, min(hb, h-y))

            cx, cy     = x + wb//2, y + hb//2
            pred_cx, pred_cy = kalman.update(cx, cy)

            # velocity
            speed_px = 0
            if prev_center:
                dx_v  = cx - prev_center[0]
                dy_v  = cy - prev_center[1]
                speed_px = np.sqrt(dx_v**2 + dy_v**2) / dt
                velocities.append(speed_px)
            prev_center = (cx, cy)
            avg_speed   = np.mean(velocities) if velocities else 0

            # trail
            trail.append((cx, cy))

            # ── DRAW ─────────────────────────────────────────

            # 1. Trajectory trail (fading dots)
            for i, (tx, ty) in enumerate(trail):
                alpha_t = int(255 * (i / len(trail)))
                radius  = max(1, int(3 * i / len(trail)))
                cv2.circle(frame, (tx, ty), radius, (alpha_t, alpha_t, 0), -1)

            # 2. Scanlines through center
            cv2.line(frame, (0, cy), (w, cy), GREY, 1)
            cv2.line(frame, (cx, 0), (cx, h), GREY, 1)

            # 3. Prediction arrow (where object is heading)
            if abs(pred_cx - cx) > 2 or abs(pred_cy - cy) > 2:
                cv2.arrowedLine(frame, (cx, cy), (pred_cx, pred_cy),
                                ORANGE, 2, tipLength=0.4)

            # 4. Bounding box + corners
            cv2.rectangle(frame, (x, y), (x+wb, y+hb), CYAN, 1)
            draw_corners(frame, x, y, wb, hb, CYAN, size=20, thickness=3)

            # 5. Center dot
            cv2.circle(frame, (cx, cy), 4, CYAN, -1)

            # 6. HUD panel (top-left of box)
            hud_x = max(0, x)
            hud_y = max(0, y - 70)
            draw_panel(frame, hud_x, hud_y, 200, 65, DARK, alpha=0.55)
            cv2.putText(frame, f"LOCK  {int(fps_display)} FPS",
                        (hud_x+6, hud_y+18), cv2.FONT_HERSHEY_PLAIN, 1.0, CYAN, 1)
            cv2.putText(frame, f"POS  X:{cx}  Y:{cy}",
                        (hud_x+6, hud_y+34), cv2.FONT_HERSHEY_PLAIN, 0.85, WHITE, 1)
            cv2.putText(frame, f"SPD  {int(avg_speed)} px/s",
                        (hud_x+6, hud_y+50), cv2.FONT_HERSHEY_PLAIN, 0.85,
                        RED if avg_speed > 300 else GREEN, 1)

            # 7. Speed bar (right edge of box)
            bar_max  = 600
            bar_h    = hb
            fill     = int(bar_h * min(avg_speed / bar_max, 1.0))
            bar_x    = x + wb + 6
            cv2.rectangle(frame, (bar_x, y),       (bar_x+8, y+bar_h), GREY, 1)
            cv2.rectangle(frame, (bar_x, y+bar_h-fill),(bar_x+8, y+bar_h),
                          RED if avg_speed > 300 else GREEN, -1)

        else:
            lost_frames += 1
            # Use Kalman to keep predicting for a few frames
            if lost_frames <= MAX_LOST and prev_center:
                pred_cx, pred_cy = kalman.update(*prev_center)
                cv2.circle(frame, (pred_cx, pred_cy), 18, ORANGE, 2)
                cv2.putText(frame, "PREDICTING...", (pred_cx-40, pred_cy-25),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, ORANGE, 1)
            else:
                is_tracking = False
                trail.clear()
                kalman.reset()
                prev_center = None
                cv2.putText(frame, "!! SIGNAL LOST — PRESS S TO REACQUIRE",
                            (20, h//2), cv2.FONT_HERSHEY_PLAIN, 1.2, RED, 2)

    # ── GLOBAL HUD ───────────────────────────────────────────
    # Top-right status bar
    draw_panel(frame, w-210, 0, 210, 30, DARK, alpha=0.6)
    status_col = GREEN if is_tracking else GREY
    cv2.putText(frame, f"KINETIC_v9  [{tracker_name}]",
                (w-205, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, status_col, 1)

    # If not tracking, show prompt
    if not is_tracking:
        draw_panel(frame, w//2-130, h//2-20, 260, 36, DARK, alpha=0.65)
        cv2.putText(frame, "PRESS [S] TO SELECT TARGET",
                    (w//2-120, h//2+8), cv2.FONT_HERSHEY_PLAIN, 1.1, CYAN, 1)

    cv2.imshow("VELOCITY_TRACKR", frame)

    # ── KEY HANDLING ─────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        ret, sel_frame = cam.read()
        if not ret:
            continue
        sel_frame = cv2.flip(sel_frame, 1)
        new_bbox  = cv2.selectROI("VELOCITY_TRACKR", sel_frame, False)
        cv2.waitKey(1)
        if new_bbox[2] > 0 and new_bbox[3] > 0:
            small_init   = cv2.resize(sel_frame,
                           (int(w*TRACK_SCALE), int(h*TRACK_SCALE)))
            scaled_bbox  = (
                int(new_bbox[0] * TRACK_SCALE),
                int(new_bbox[1] * TRACK_SCALE),
                int(new_bbox[2] * TRACK_SCALE),
                int(new_bbox[3] * TRACK_SCALE),
            )
            tracker, tracker_name = make_tracker()
            tracker.init(small_init, scaled_bbox)
            is_tracking = True
            trail.clear()
            velocities.clear()
            kalman.reset()
            prev_center = None
            lost_frames = 0

    elif key == ord("r"):
        is_tracking = False
        tracker, tracker_name = make_tracker()
        trail.clear()
        velocities.clear()
        kalman.reset()
        prev_center = None

    elif key == ord("t"):
        # Cycle through tracker types
        tracker_idx  = (tracker_idx + 1) % len(tracker_names_list)
        next_name    = tracker_names_list[tracker_idx]
        try:
            tracker      = getattr(cv2, f"Tracker{next_name}_create")()
            tracker_name = next_name
            is_tracking  = False
            print(f"Switched to: {tracker_name}")
        except AttributeError:
            print(f"{next_name} not available, skipping")

    elif key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()