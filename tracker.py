import cv2
import time
import threading
import numpy as np
from collections import deque
import os

# ================================================================
#  VELOCITY_TRACKR v10 — ULTIMATE EDITION
#  - Click to select target (no S key needed)
#  - Multi-object tracking (up to 3 targets)
#  - Auto re-acquire on signal loss
#  - Face/body detection assist
#  - Zoom window on target
#  - Recording feature
#  - Heat map trail
#  - Radar mini-map
#  - Full neon military HUD
# ================================================================

# ── Kalman Filter ───────────────────────────────────────────────
class KalmanPredictor:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, cx, cy):
        cx, cy = int(cx), int(cy)
        m = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self.initialized:
            self.kf.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
            self.initialized = True
        self.kf.correct(m)
        pred = self.kf.predict()
        return int(pred[0].item()), int(pred[1].item())

    def reset(self):
        self.initialized = False


# ── Target object (one per tracked thing) ───────────────────────
class Target:
    COLORS = [
        (255, 220, 0),    # cyan-yellow
        (0, 200, 255),    # orange
        (180, 0, 255),    # purple
    ]

    def __init__(self, idx, tracker, bbox, frame, scale, label="TGT"):
        self.idx         = idx
        self.tracker     = tracker
        self.kalman      = KalmanPredictor()
        self.trail       = deque(maxlen=60)
        self.velocities  = deque(maxlen=15)
        self.prev_center = None
        self.lost_frames = 0
        self.active      = True
        self.color       = self.COLORS[idx % len(self.COLORS)]
        self.label       = f"{label}-{idx+1}"
        self.scale       = scale
        self.bbox        = bbox       # full-res bbox
        self.heatmap     = None

        # init tracker on scaled frame
        sh, sw = frame.shape[:2]
        small  = cv2.resize(frame, (int(sw*scale), int(sh*scale)))
        sb     = (int(bbox[0]*scale), int(bbox[1]*scale),
                  int(bbox[2]*scale), int(bbox[3]*scale))
        self.tracker.init(small, sb)

    def update(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w*self.scale), int(h*self.scale)))
        ok, sb = self.tracker.update(small)
        if ok:
            self.lost_frames = 0
            s  = 1 / self.scale
            x  = max(0, int(sb[0]*s))
            y  = max(0, int(sb[1]*s))
            wb = max(1, min(int(sb[2]*s), w-x))
            hb = max(1, min(int(sb[3]*s), h-y))
            self.bbox = (x, y, wb, hb)
            cx, cy = x + wb//2, y + hb//2
            self.kalman.update(cx, cy)
            if self.prev_center:
                dx = cx - self.prev_center[0]
                dy = cy - self.prev_center[1]
                self.velocities.append(np.sqrt(dx**2 + dy**2))
            self.prev_center = (cx, cy)
            self.trail.append((cx, cy))
            return True
        else:
            self.lost_frames += 1
            return False

    @property
    def center(self):
        if self.bbox:
            x, y, wb, hb = self.bbox
            return x + wb//2, y + hb//2
        return None

    @property
    def speed(self):
        return float(np.mean(self.velocities)) if self.velocities else 0


# ── Threaded camera ─────────────────────────────────────────────
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FPS,         120)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
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
            return (True, self.frame.copy()) if self.ret else (False, None)

    def release(self):
        self.running = False
        self.cap.release()


# ── Tracker factory ─────────────────────────────────────────────
TRACKER_TYPES = ["CSRT", "KCF", "MOSSE"]
tracker_idx   = 0

def make_tracker():
    name = TRACKER_TYPES[tracker_idx]
    try:
        return getattr(cv2, f"Tracker{name}_create")(), name
    except AttributeError:
        return cv2.TrackerKCF_create(), "KCF"


# ── Drawing helpers ─────────────────────────────────────────────
def draw_panel(frame, x, y, w, h, color=(10,10,10), alpha=0.5):
    x, y, w, h = int(x), int(y), int(w), int(h)
    x2, y2 = min(x+w, frame.shape[1]), min(y+h, frame.shape[0])
    x,  y  = max(0, x), max(0, y)
    if x2 <= x or y2 <= y:
        return
    roi = frame[y:y2, x:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0,0), (x2-x, y2-y), color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, roi)
    frame[y:y2, x:x2] = roi
    cv2.rectangle(frame, (x, y), (x2, y2), color, 1)

def draw_corners(frame, x, y, wb, hb, color, size=22, thick=3):
    for px, py, dx, dy in [
        (x,    y,    1,  1), (x+wb, y,   -1,  1),
        (x,    y+hb, 1, -1), (x+wb, y+hb,-1, -1),
    ]:
        cv2.line(frame, (px, py), (px+dx*size, py),        color, thick)
        cv2.line(frame, (px, py), (px,          py+dy*size),color, thick)

def draw_dashed_rect(frame, x, y, wb, hb, color, gap=8):
    pts = [(x+i, y) for i in range(0, wb, gap*2)] + \
          [(x+wb, y+i) for i in range(0, hb, gap*2)] + \
          [(x+wb-i, y+hb) for i in range(0, wb, gap*2)] + \
          [(x, y+hb-i) for i in range(0, hb, gap*2)]
    for i in range(0, len(pts)-1, 2):
        cv2.line(frame, pts[i], pts[i+1], color, 1)

def draw_zoom_window(frame, bbox, size=120):
    x, y, wb, hb = [int(v) for v in bbox]
    pad = 10
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(frame.shape[1], x+wb+pad), min(frame.shape[0], y+hb+pad)
    if x2 <= x1 or y2 <= y1:
        return
    crop   = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(crop, (size, size))
    # neon border
    cv2.rectangle(zoomed, (0,0), (size-1,size-1), (255,220,0), 2)
    # crosshair
    cv2.line(zoomed, (size//2, 0), (size//2, size), (50,50,50), 1)
    cv2.line(zoomed, (0, size//2), (size, size//2), (50,50,50), 1)
    cv2.circle(zoomed, (size//2, size//2), 6, (0,255,120), 1)
    # paste into top-right corner
    fw = frame.shape[1]
    frame[10:10+size, fw-size-10:fw-10] = zoomed
    draw_panel(frame, fw-size-12, 8, size+4, size+4, (20,20,20), 0.3)

def draw_radar(frame, targets, w, h, size=90):
    rx, ry = w - size - 10, h - size - 10
    draw_panel(frame, rx, ry, size, size, (5,15,5), 0.65)
    cx2, cy2 = rx + size//2, ry + size//2
    # grid circles
    for r in [size//6, size//3, size//2-2]:
        cv2.circle(frame, (cx2, cy2), r, (0,60,0), 1)
    cv2.line(frame, (rx, cy2), (rx+size, cy2), (0,40,0), 1)
    cv2.line(frame, (cx2, ry), (cx2, ry+size), (0,40,0), 1)
    cv2.putText(frame, "RADAR", (rx+4, ry+10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,120,0), 1)
    # plot targets
    for t in targets:
        if t.center:
            tx = int(rx + (t.center[0] / w) * size)
            ty = int(ry + (t.center[1] / h) * size)
            tx = max(rx+2, min(rx+size-2, tx))
            ty = max(ry+2, min(ry+size-2, ty))
            cv2.circle(frame, (tx, ty), 4, t.color, -1)
            cv2.circle(frame, (tx, ty), 7, t.color, 1)

def draw_scanline_overlay(frame, tick):
    h = frame.shape[0]
    y = int((tick * 3) % h)
    cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 0), 1)
    overlay = frame.copy()
    for row in range(0, h, 4):
        cv2.line(overlay, (0, row), (frame.shape[1], row), (0,0,0), 1)
    cv2.addWeighted(overlay, 0.04, frame, 0.96, 0, frame)


# ── Face detector for auto-detect assist ────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
    return [(x, y, w, h) for (x, y, w, h) in faces] if len(faces) else []


# ── Mouse click handler ─────────────────────────────────────────
click_point  = None
click_active = False

def on_mouse(event, x, y, flags, param):
    global click_point, click_active
    if event == cv2.EVENT_LBUTTONDBLCLK:
        click_point  = (x, y)
        click_active = True


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
cam = CameraStream(0)
time.sleep(0.6)

WIN = "VELOCITY_TRACKR  v10"
cv2.namedWindow(WIN)
cv2.setMouseCallback(WIN, on_mouse)

targets      = []          # list of Target objects
MAX_TARGETS  = 3
TRACK_SCALE  = 0.6
recording    = False
writer       = None
show_zoom    = True
show_radar   = True
show_faces   = False
auto_reacq   = True        # auto re-acquire lost targets
MAX_LOST     = 12
tick         = 0
prev_time    = time.time()
fps_display  = 0

_, tracker_name = make_tracker()

print("╔══════════════════════════════════════╗")
print("║     VELOCITY_TRACKR  v10             ║")
print("╠══════════════════════════════════════╣")
print("║  [DBL-CLICK]  Add target             ║")
print("║  [S]          Manual ROI select      ║")
print("║  [R]          Clear all targets      ║")
print("║  [T]          Cycle tracker type     ║")
print("║  [F]          Toggle face detect     ║")
print("║  [Z]          Toggle zoom window     ║")
print("║  [V]          Start/stop recording   ║")
print("║  [Q]          Quit                   ║")
print("╚══════════════════════════════════════╝")

while True:
    ok, frame = cam.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    tick += 1

    now         = time.time()
    dt          = max(now - prev_time, 1e-6)
    fps_display = 0.85 * fps_display + 0.15 * (1/dt)
    prev_time   = now

    # ── FACE DETECTION (every 10 frames) ──────────────────────
    if show_faces and tick % 10 == 0:
        faces = detect_faces(frame)
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0,80,0), 1)
            cv2.putText(frame, "FACE", (fx, fy-5),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, (0,120,0), 1)

    # ── DOUBLE-CLICK TO ADD TARGET ─────────────────────────────
    if click_active and click_point:
        click_active = False
        if len(targets) < MAX_TARGETS:
            cx_c, cy_c = click_point
            # make a bbox around click point
            bw, bh = 80, 80
            bx = max(0, cx_c - bw//2)
            by = max(0, cy_c - bh//2)
            bx = min(bx, w - bw)
            by = min(by, h - bh)
            tr, tname = make_tracker()
            t = Target(len(targets), tr, (bx, by, bw, bh), frame, TRACK_SCALE)
            targets.append(t)
            print(f"Target {len(targets)} locked [{tname}]")

    # ── UPDATE ALL TARGETS ─────────────────────────────────────
    for t in targets[:]:
        if not t.active:
            continue
        ok_t = t.update(frame)
        if not ok_t:
            if t.lost_frames > MAX_LOST:
                if auto_reacq and t.prev_center:
                    # Try to re-init tracker around last known position
                    lx, ly = t.prev_center
                    bw, bh = t.bbox[2], t.bbox[3]
                    bx = max(0, lx - bw//2)
                    by = max(0, ly - bh//2)
                    new_tr, _ = make_tracker()
                    try:
                        t.tracker = new_tr
                        small_f   = cv2.resize(frame, (int(w*TRACK_SCALE), int(h*TRACK_SCALE)))
                        sb        = (int(bx*TRACK_SCALE), int(by*TRACK_SCALE),
                                     int(bw*TRACK_SCALE), int(bh*TRACK_SCALE))
                        t.tracker.init(small_f, sb)
                        t.lost_frames = 0
                    except:
                        t.active = False
                else:
                    t.active = False

    # Remove dead targets
    targets = [t for t in targets if t.active]

    # ── DRAW TARGETS ───────────────────────────────────────────
    for t in targets:
        if not t.center:
            continue
        x, y, wb, hb = [int(v) for v in t.bbox]
        cx, cy       = t.center
        col          = t.color
        spd          = t.speed

        # trail
        trail_list = list(t.trail)
        for i in range(1, len(trail_list)):
            alpha_t  = i / len(trail_list)
            r        = max(1, int(4 * alpha_t))
            shade    = tuple(int(c * alpha_t) for c in col)
            cv2.circle(frame, trail_list[i], r, shade, -1)

        # scanlines through center
        cv2.line(frame, (0, cy), (w, cy), (40,40,40), 1)
        cv2.line(frame, (cx, 0), (cx, h), (40,40,40), 1)

        # prediction arrow
        if t.kalman.initialized:
            px, py = t.kalman.update(cx, cy)
            if abs(px-cx) > 3 or abs(py-cy) > 3:
                cv2.arrowedLine(frame, (cx,cy), (px,py), (0,165,255), 2, tipLength=0.35)

        # dashed outer rect
        draw_dashed_rect(frame, x-4, y-4, wb+8, hb+8, col, gap=6)
        # solid inner rect
        cv2.rectangle(frame, (x, y), (x+wb, y+hb), col, 1)
        # corners
        draw_corners(frame, x, y, wb, hb, col, size=22, thick=3)
        # center dot
        cv2.circle(frame, (cx, cy), 5, col, -1)
        cv2.circle(frame, (cx, cy), 9, col, 1)

        # HUD panel
        hx = max(0, x)
        hy = max(10, y - 75)
        draw_panel(frame, hx, hy, 220, 70, (5,5,5), 0.6)
        cv2.putText(frame, f"{t.label}  {int(fps_display)}FPS",
                    (hx+6, hy+16), cv2.FONT_HERSHEY_PLAIN, 1.0, col, 1)
        cv2.putText(frame, f"POS  {cx} , {cy}",
                    (hx+6, hy+32), cv2.FONT_HERSHEY_PLAIN, 0.85, (200,200,200), 1)
        cv2.putText(frame, f"SPD  {int(spd*30)} px/s",
                    (hx+6, hy+48), cv2.FONT_HERSHEY_PLAIN, 0.85,
                    (0,60,255) if spd > 10 else (0,255,120), 1)
        cv2.putText(frame, f"LOCK {t.lost_frames == 0 and 'SOLID' or 'WEAK'}",
                    (hx+6, hy+64), cv2.FONT_HERSHEY_PLAIN, 0.85,
                    (0,255,120) if t.lost_frames == 0 else (0,165,255), 1)

        # speed bar
        bar_max = 20
        fill    = int(hb * min(spd / bar_max, 1.0))
        bx2     = x + wb + 8
        cv2.rectangle(frame, (bx2, y),          (bx2+8, y+hb), (50,50,50), 1)
        cv2.rectangle(frame, (bx2, y+hb-fill),  (bx2+8, y+hb),
                      (0,60,255) if spd > 10 else (0,255,120), -1)

        # zoom window (first target only)
        if show_zoom and t.idx == 0:
            draw_zoom_window(frame, t.bbox, size=110)

    # ── PREDICTING overlay for lost targets ────────────────────
    for t in targets:
        if t.lost_frames > 0 and t.prev_center:
            px, py = t.kalman.update(*t.prev_center)
            cv2.circle(frame, (px, py), 20, (0,165,255), 2)
            cv2.putText(frame, f"REACQUIRING {t.label}",
                        (px-50, py-28), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,165,255), 1)

    # ── RADAR ──────────────────────────────────────────────────
    if show_radar and targets:
        draw_radar(frame, targets, w, h, size=100)

    # ── CRT SCANLINE FX ────────────────────────────────────────
    draw_scanline_overlay(frame, tick)

    # ── TOP STATUS BAR ─────────────────────────────────────────
    draw_panel(frame, 0, 0, w, 28, (5,5,5), 0.7)
    trk_col = (0,255,120) if targets else (70,70,70)
    cv2.putText(frame, f"VELOCITY_TRACKR v10  [{tracker_name}]  "
                        f"TGT:{len(targets)}/{MAX_TARGETS}  "
                        f"{int(fps_display)}FPS  "
                        f"{'[REC]' if recording else ''}",
                (8, 18), cv2.FONT_HERSHEY_PLAIN, 1.0, trk_col, 1)

    # ── BOTTOM HINT BAR ────────────────────────────────────────
    if not targets:
        draw_panel(frame, 0, h-30, w, 30, (5,5,5), 0.65)
        cv2.putText(frame, "DBL-CLICK to lock target   [S] ROI select   [F] face detect   [Q] quit",
                    (8, h-10), cv2.FONT_HERSHEY_PLAIN, 0.9, (100,100,100), 1)

    # ── RECORDING ──────────────────────────────────────────────
    if recording:
        if writer is None:
            ts     = time.strftime("%Y%m%d_%H%M%S")
            fname  = f"TRACKR_{ts}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(fname, fourcc, 30, (w, h))
            print(f"Recording: {fname}")
        writer.write(frame)
        # red dot
        cv2.circle(frame, (w-20, 14), 6, (0,0,255), -1)

    cv2.imshow(WIN, frame)

    # ── KEYS ───────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        # Manual ROI
        ok2, sf = cam.read()
        if ok2:
            sf       = cv2.flip(sf, 1)
            new_bbox = cv2.selectROI(WIN, sf, False)
            cv2.waitKey(1)
            if new_bbox[2] > 0 and new_bbox[3] > 0 and len(targets) < MAX_TARGETS:
                tr, tname = make_tracker()
                t = Target(len(targets), tr, new_bbox, sf, TRACK_SCALE)
                targets.append(t)

    elif key == ord("r"):
        targets.clear()
        print("All targets cleared")

    elif key == ord("t"):
        tracker_idx = (tracker_idx + 1) % len(TRACKER_TYPES)
        _, tracker_name = make_tracker()
        targets.clear()
        print(f"Tracker: {tracker_name} — re-select targets")

    elif key == ord("f"):
        show_faces = not show_faces
        print(f"Face detect: {'ON' if show_faces else 'OFF'}")

    elif key == ord("z"):
        show_zoom = not show_zoom

    elif key == ord("v"):
        recording = not recording
        if not recording and writer:
            writer.release()
            writer = None
            print("Recording saved")

    elif key == ord("q"):
        break

if writer:
    writer.release()
cam.release()
cv2.destroyAllWindows()