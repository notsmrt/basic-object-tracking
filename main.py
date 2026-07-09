import cv2
import json
import math
import time
import threading
import queue
from collections import deque
from datetime import datetime
import numpy as np
from ultralytics import YOLO

# =============================================================================
# Configuration
# =============================================================================
HEADLESS_ENABLED = False
SAVE_VIDEO_ENABLED = False
OUTPUT_VIDEO_PATH = "output.mp4"
CLAHE_ENABLED = False

# Source: camera index (int) OR a video-file path (str) for offline replay/testing
SOURCE = 0
CAMERA_WIDTH = 0
CAMERA_HEIGHT = 0
CAMERA_FPS = 0

# Detection (runs in its own thread, as fast as it can — "detect slow")
YOLO_WEIGHTS = "yolov8n.pt"       # -> yolov8m.pt (or larger) on the GPU machine
IMG_SIZE = 640                    # bump to 1280 for small-object recall
CONFIDENCE_THRESHOLD = 0.5
YOLO_NMS_IOU = 0.45

# BoT-SORT association + appearance ReID + (optional) camera motion compensation
TRACKER_RUNTIME_CFG = "botsort_runtime.yaml"  # written at startup from the knobs below
BOTSORT_WITH_REID = True          # appearance ReID: "it's the same phone that moved"
GMC_METHOD = "none"               # static camera now; set "sparseOptFlow" when SLAM/motion added
BOTSORT_TRACK_BUFFER = 90         # detector frames a lost track is kept for re-association

# --- Fast tracker: per-frame optical flow ("general pixels") ---
OPTFLOW_MAX_POINTS = 30
OPTFLOW_QUALITY = 0.3
OPTFLOW_MIN_POINTS = 5             # below this, flow is considered lost -> pure Kalman coast
OPTFLOW_MEASUREMENT_R_SCALE = 4.0 # flow is a weaker measurement than a real detection
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# --- Phantom / predicted tracks ---
TRACKED_HOLD_S = 0.30             # after a detection, stay "TRACKED" this long before "PREDICTED"
COAST_MAX_S = 4.0                 # keep predicting a missing (non-occluded) object at most this
                                   # long; was 1.5s -> too eager to retire real, still-present
                                   # objects the detector just hasn't re-picked-up yet
CONF_DECAY_PER_S = 0.35           # track_confidence *= exp(-rate*dt) while undetected; slowed so
                                   # confidence doesn't hit CONF_FLOOR before COAST_MAX_S elapses
CONF_FLOOR = 0.15                 # retire below this confidence

# --- Identity memory / re-association bridge (broad pickup) ---
REID_MEMORY_S = 6.0                # keep a lost object's signature this long; matches the
                                    # longer COAST_MAX_S below so re-acquisition window isn't
                                    # the bottleneck once coasting was extended
# Re-acquisition uses a search radius around the predicted position that GROWS with the
# object's speed and how long it's been lost, so fast movers can be picked back up.
REID_BASE_RADIUS_NORM = 0.10      # base search radius as a fraction of the frame diagonal
REID_HIST_SIM_BROAD = 0.30        # min HSV color-histogram correlation to re-claim identity
OCCLUSION_IOU_THRESHOLD = 0.1     # min IOU with another track to be considered occluded

# Keep stationary objects that vanish in the interior of the scene (occlusion / detector
# flicker) instead of retiring them. Objects lost near a frame edge are assumed to have left.
EDGE_MARGIN_PX = 25               # a box within this many px of any edge counts as "at the edge"
DORMANT_MAX_S = 5.0               # absolute cap on holding a dormant object (0 = no cap)
# A track only becomes DORMANT if it was a real, established object (high confidence, seen
# repeatedly), was settled (stationary) when last seen, and still visibly occupies its spot.
DORMANT_MIN_CONF = 0.6            # peak detection confidence required to qualify as dormant
DORMANT_MIN_DETECTIONS = 5       # must have been detected at least this many times
DORMANT_PRESENCE_SIM = 0.4       # region must still look like the object (HSV hist corr)
DORMANT_WAKE_PX = 6.0            # per-frame optical-flow motion that wakes a dormant object
DORMANT_MISSING_MAX = 60         # frames a dormant object may look absent before it is retired
FLOW_MATCH_SIM = 0.4            # min appearance match to trust optical flow (rejects occluders)

# --- Temporal / kinetic motion model ---
VELOCITY_SMOOTHING = 0.3          # less EMA lag -> speed responds faster
VELOCITY_INJECT = 0.6             # blend of detection-measured velocity injected into the KF
MOTION_SPEED_THRESHOLD_NORM = 0.03
MOTION_ARROW_LOOKAHEAD_S = 0.4

# Kalman filter (per track) — constant-acceleration with adaptive acceleration noise
KALMAN_PROCESS_NOISE = 1e-2           # position + size process noise
KALMAN_VELOCITY_PROCESS_NOISE = 5e-2  # velocity process noise (accel now carries fast changes)
KALMAN_ACCEL_PROCESS_NOISE = 5e-2     # BASE acceleration noise, scaled adaptively per frame
KALMAN_MEASUREMENT_NOISE = 5e-1
KALMAN_ADAPT_ALPHA = 0.3              # EMA rate for the adaptive acceleration-noise multiplier
KALMAN_INNOV_EMA = 0.05              # slow baseline of innovation; q_scale = innov / baseline
KALMAN_QSCALE_MIN = 0.5              # floor on the acceleration-noise multiplier (steady motion)
KALMAN_QSCALE_MAX = 25.0             # ceiling (sharp accel/decel) -> fast adaptation, no overshoot
ACCEL_COAST_DECAY = 0.8             # per-frame acceleration decay once detections/flow stop
VELOCITY_COAST_DECAY = 0.85        # per-frame velocity decay once we can't see the object -> a
                                    # lost object settles in place instead of flying off-screen
COAST_DECAY_GRACE_S = 0.1          # start decaying this soon after the last correction (flow-locked
                                    # objects correct every frame, so only invisible ones decay)
KALMAN_MAX_DT = 0.2                 # cap on predict dt (s) so a stale/revived track can't explode

# SAM2
SAM2_WEIGHTS = "sam2.1_t.pt"
SAM_ENABLED = True                # auto-disabled if no CUDA
SAM_AUTO_DISCOVERY_ENABLED = False
SAM_AUTOMATIC_EVERY_N_FRAMES = 30

# Depth Anything V2 (metric, indoor/Hypersim checkpoint -> scene-consistent scale so
# depth is comparable frame-to-frame, not just relative within one frame). Used to break
# ties on overlapping boxes: whichever track is farther from the camera is the one that's
# actually occluded, instead of guessing symmetrically off IOU alone.
DEPTH_ENABLED = True              # auto-disabled if no CUDA, same gating as SAM2
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small-hf"
DEPTH_INFER_SIZE = 384            # downscaled inference size; map is upsampled back to frame size
DEPTH_EVERY_N_FRAMES = 10         # matches SAM's broad-pass rhythm (SAM_AUTOMATIC_EVERY_N_FRAMES) -
                                   # occlusion ordering / hijack detection / re-acquisition all
                                   # operate on object-crossing timescales, not single frames, so a
                                   # ~10-frame-stale map is fine and it cuts GPU contention with YOLO+SAM
DEPTH_HIJACK_DELTA_M = 0.35       # min depth discontinuity under a flow-advected box to call it a
                                   # hijack (an occluder crossing the box sits at a different depth
                                   # than the tracked object) -- tuned for indoor/Hypersim room scale
REID_DEPTH_TOL_M = 0.6            # max depth mismatch allowed between a re-acquisition candidate and
                                   # a lost track's last-known depth; looser than the hijack threshold
                                   # since more time (up to REID_MEMORY_S) may have passed

# Output logging
LOG_ENABLED = True
LOG_PATH = "perception_log.jsonl"
PRINT_SUMMARY = True
# Bump whenever a field is added/removed/renamed in build_records or the top-level log record,
# so a downstream parser/LLM consuming perception_log.jsonl can detect a schema change instead
# of silently misreading fields.
LOG_SCHEMA_VERSION = 2
CONTOUR_SIMPLIFY_EPS = 0.01
TRAJECTORY_LEN = 30

# Debug: artificially drop detection windows to exercise phantom prediction (0 = off)
DEBUG_DROP_DETECTIONS_EVERY_N = 0

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)]
DISCOVERED_COLOR = (170, 170, 170)

# =============================================================================
# Queues
# =============================================================================
det_in_q = queue.Queue(maxsize=1)       # main -> detection : (frame, id, ts)
det_out_q = queue.Queue(maxsize=1)      # detection -> main : (id, ts, [(tid,bbox,conf,cls), ...])
sam_in_q = queue.Queue(maxsize=1)       # main -> sam  : (frame, id, tracks)
sam_masks_q = queue.Queue(maxsize=1)    # sam  -> main : [mask, ...]
sam_feedback_q = queue.Queue(maxsize=1) # sam  -> main : {object_id: {refined...}}
depth_in_q = queue.Queue(maxsize=1)     # main -> depth : (frame, id)
depth_out_q = queue.Queue(maxsize=1)    # depth -> main : (id, depth_map[h,w] float32 meters)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Geometry / image helpers
# =============================================================================
def apply_clahe(frame):
    if not CLAHE_ENABLED:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    limg = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def safe_num(v, nd=4):
    """Round for logging, collapsing NaN/Inf to None -> a downstream JSON parser (or an LLM
    reading the log) never has to special-case a non-finite value; a divergent Kalman state
    (e.g. from a stale/revived track or a bad dt) fails closed instead of poisoning the record."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def clamp_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1 = int(min(max(x1, 0), w - 1))
    y1 = int(min(max(y1, 0), h - 1))
    x2 = int(min(max(x2, 0), w - 1))
    y2 = int(min(max(y2, 0), h - 1))
    if x2 <= x1:
        x2 = min(x1 + 1, w - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, h - 1)
    return (x1, y1, x2, y2)


def calculate_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    denom = float(areaA + areaB - inter)
    return inter / denom if denom > 0 else 0.0


def direction_label(vx, vy):
    ang = math.degrees(math.atan2(-vy, vx)) % 360.0
    dirs = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    return dirs[int((ang + 22.5) // 45) % 8], round(ang, 1)


def at_edge(bbox, w, h, margin=EDGE_MARGIN_PX):
    """True if the box touches the frame border (object likely leaving/left the scene)."""
    x1, y1, x2, y2 = bbox
    return x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin


def resize_mask(seg, w, h):
    if seg.shape[:2] == (h, w):
        return seg
    return cv2.resize(seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def color_hist(frame, bbox):
    x1, y1, x2, y2 = clamp_bbox(bbox, frame.shape[1], frame.shape[0])
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def hist_sim(a, b):
    if a is None or b is None:
        return 0.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def track_depth(depth_map, bbox):
    """Median metric depth over a track's box, shrunk slightly so edge pixels that belong
    to the background (or a neighboring, overlapping object) don't skew the estimate."""
    if depth_map is None:
        return None
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None
    mx, my = max(1, int(bw * 0.15)), max(1, int(bh * 0.15))
    ix1, iy1 = min(x1 + mx, x2 - 1), min(y1 + my, y2 - 1)
    ix2, iy2 = max(x2 - mx, ix1 + 1), max(y2 - my, iy1 + 1)
    roi = depth_map[iy1:iy2, ix1:ix2]
    if roi.size == 0:
        return None
    return float(np.median(roi))


# =============================================================================
# Optical flow ("general pixels" — follow an object between detections)
# =============================================================================
def seed_points(gray, bbox):
    x1, y1, x2, y2 = clamp_bbox(bbox, gray.shape[1], gray.shape[0])
    roi = gray[y1:y2, x1:x2]
    if roi.shape[0] < 3 or roi.shape[1] < 3:
        return None
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=OPTFLOW_MAX_POINTS,
                                  qualityLevel=OPTFLOW_QUALITY, minDistance=5)
    if pts is None:
        return None
    pts = pts.astype(np.float32)
    pts[:, 0, 0] += x1
    pts[:, 0, 1] += y1
    return pts


def flow_step(prev_gray, cur_gray, pts):
    """Advect feature points by Lucas-Kanade; return (new_pts, median_dx, median_dy, ok)."""
    if pts is None or len(pts) < 1:
        return None, 0.0, 0.0, False
    new_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, **LK_PARAMS)
    if new_pts is None:
        return None, 0.0, 0.0, False
    st = st.ravel().astype(bool)
    good_new, good_old = new_pts[st], pts[st]
    if len(good_new) < OPTFLOW_MIN_POINTS:
        return None, 0.0, 0.0, False
    delta = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
    return good_new.reshape(-1, 1, 2), float(np.median(delta[:, 0])), float(np.median(delta[:, 1])), True


# =============================================================================
# Per-track Kalman filter: constant ACCELERATION on [cx, cy, w, h], with adaptive
# acceleration (process) noise. The acceleration states let the filter anticipate
# deceleration (less overshoot/undershoot); the process noise on those states grows
# when the measurement disagrees with the prediction (motion changing) and shrinks when
# motion is steady -> faster response to accel changes + steadier velocity direction.
# =============================================================================
class KalmanBoxTracker:
    def __init__(self, bbox, now):
        x1, y1, x2, y2 = bbox
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        kf = cv2.KalmanFilter(8, 4)   # state [cx,cy,w,h,vx,vy,ax,ay]; measure [cx,cy,w,h]
        kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self._R = np.eye(4, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        kf.measurementNoiseCov = self._R.copy()
        kf.errorCovPost = np.eye(8, dtype=np.float32)
        kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], np.float32)
        self.kf = kf
        self.last_t = now
        self.last_correct_t = now
        self.q_scale = 1.0            # adaptive acceleration-noise multiplier (innovation-driven)
        self.innov_ema = None         # slow baseline of innovation, for self-normalized adaptation
        self.recent_ratios = deque(maxlen=2)  # last two innovation/baseline ratios -> onset detector

    def predict(self, now):
        dt = min(max(now - self.last_t, 1e-3), KALMAN_MAX_DT)  # cap so a stale track can't blow up
        self.last_t = now
        # Once detections/flow stop we no longer actually observe the object, so bleed off both
        # velocity and acceleration: a lost object settles in place rather than dead-reckoning
        # off-screen. (A still-visible object keeps getting flow corrections, so it won't decay.)
        if now - self.last_correct_t > COAST_DECAY_GRACE_S:
            self.kf.statePost[4, 0] *= VELOCITY_COAST_DECAY
            self.kf.statePost[5, 0] *= VELOCITY_COAST_DECAY
            self.kf.statePost[6, 0] *= ACCEL_COAST_DECAY
            self.kf.statePost[7, 0] *= ACCEL_COAST_DECAY
        T = np.eye(8, dtype=np.float32)
        T[0, 4] = T[1, 5] = dt                    # cx += vx*dt, cy += vy*dt
        T[4, 6] = T[5, 7] = dt                    # vx += ax*dt, vy += ay*dt
        T[0, 6] = T[1, 7] = 0.5 * dt * dt         # cx += 0.5*ax*dt^2 (anticipates accel)
        self.kf.transitionMatrix = T
        q = np.array([KALMAN_PROCESS_NOISE] * 4 + [KALMAN_VELOCITY_PROCESS_NOISE] * 2 +
                     [KALMAN_ACCEL_PROCESS_NOISE * self.q_scale] * 2, np.float32)
        self.kf.processNoiseCov = np.diag(q)
        self.kf.predict()

    def correct(self, bbox, r_scale=1.0):
        x1, y1, x2, y2 = bbox
        m = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2], [x2 - x1], [y2 - y1]], np.float32)
        # Innovation = how far the measurement is from the a-priori prediction. Compared against
        # its own slow baseline: when it spikes (unmodeled accel/decel), raise the acceleration
        # noise so the filter adapts fast; when steady, q_scale settles near 1.
        pre = self.kf.statePre.ravel()
        innov = math.hypot(float(m[0, 0]) - pre[0], float(m[1, 0]) - pre[1])
        if self.innov_ema is None:
            self.innov_ema = innov
        ratio = innov / (self.innov_ema + 1.0)   # +1px desensitizes sub-pixel jitter
        target = min(max(ratio, KALMAN_QSCALE_MIN), KALMAN_QSCALE_MAX)
        # Motion-onset fast path: from rest, the EMA-eased ratio above takes a few frames to
        # ramp up because early innovation is small in absolute terms even though it's a real,
        # consistent break from stillness. If the last two corrections both show a clear,
        # above-baseline disagreement (not one noisy frame), snap straight to max acceleration
        # noise instead of easing there -> the filter picks up "it just started moving"
        # immediately rather than lagging the actual onset of acceleration.
        self.recent_ratios.append(ratio)
        if len(self.recent_ratios) == self.recent_ratios.maxlen and min(self.recent_ratios) > 2.0:
            self.q_scale = KALMAN_QSCALE_MAX
        else:
            self.q_scale = (1 - KALMAN_ADAPT_ALPHA) * self.q_scale + KALMAN_ADAPT_ALPHA * target
        self.innov_ema = (1 - KALMAN_INNOV_EMA) * self.innov_ema + KALMAN_INNOV_EMA * innov
        self.kf.measurementNoiseCov = self._R * r_scale
        self.kf.correct(m)
        self.last_correct_t = self.last_t

    def set_velocity(self, vx, vy, blend):
        """Nudge the state's velocity toward a directly-measured value (kills warm-up lag)."""
        self.kf.statePost[4, 0] = (1 - blend) * float(self.kf.statePost[4, 0]) + blend * vx
        self.kf.statePost[5, 0] = (1 - blend) * float(self.kf.statePost[5, 0]) + blend * vy

    def state(self):
        s = self.kf.statePost.ravel()
        cx, cy, w, h, vx, vy = s[0], s[1], s[2], s[3], s[4], s[5]
        w, h = max(w, 1.0), max(h, 1.0)
        return ((int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)),
                (int(cx), int(cy)), (float(vx), float(vy)))


def apply_motion(track, bbox, center, vel, diag):
    pvx, pvy = track.get('velocity', (0.0, 0.0))
    vx = pvx * VELOCITY_SMOOTHING + vel[0] * (1 - VELOCITY_SMOOTHING)
    vy = pvy * VELOCITY_SMOOTHING + vel[1] * (1 - VELOCITY_SMOOTHING)
    speed = math.hypot(vx, vy)
    sn = speed / diag
    moving = sn > MOTION_SPEED_THRESHOLD_NORM
    d_lab, d_deg = direction_label(vx, vy)
    track.update({
        'bbox': bbox, 'bbox_center': center,
        'velocity': (vx, vy), 'speed_px_s': speed, 'speed_norm': sn,
        'direction': d_lab if moving else '-', 'direction_deg': d_deg,
        'motion_state': 'moving' if moving else 'stationary',
    })


# =============================================================================
# Detection thread: YOLO + BoT-SORT (ReID) -> raw (tracker_id, bbox, conf, cls)
# =============================================================================
def write_tracker_cfg(path, with_reid):
    lines = [
        "tracker_type: botsort",
        "track_high_thresh: 0.25", "track_low_thresh: 0.1", "new_track_thresh: 0.25",
        f"track_buffer: {BOTSORT_TRACK_BUFFER}", "match_thresh: 0.8", "fuse_score: True",
        f"gmc_method: {GMC_METHOD}",
        "proximity_thresh: 0.5", "appearance_thresh: 0.8",
        f"with_reid: {with_reid}", "model: auto",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def detection_worker(yolo_model):
    while True:
        frame, frame_id, ts = det_in_q.get()
        if frame is None:
            break
        results = yolo_model.track(frame, persist=True, tracker=TRACKER_RUNTIME_CFG,
                                   conf=CONFIDENCE_THRESHOLD, iou=YOLO_NMS_IOU,
                                   imgsz=IMG_SIZE, verbose=False)
        dets = []
        drop = (DEBUG_DROP_DETECTIONS_EVERY_N and
                (frame_id % (2 * DEBUG_DROP_DETECTIONS_EVERY_N)) >= DEBUG_DROP_DETECTIONS_EVERY_N)
        boxes = results[0].boxes
        if not drop and boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.int().cpu().tolist()
            for tid, box, cf, cl in zip(ids, xyxy, confs, clss):
                dets.append((tid, tuple(map(int, box)), float(cf), yolo_model.names[cl]))
        if det_out_q.empty():
            det_out_q.put((frame_id, ts, dets))


# =============================================================================
# SAM2 thread (prompted by the fast tracker's current boxes)
# =============================================================================
def sam_worker(weights):
    import torch
    from ultralytics import SAM
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Loading SAM2 ({weights}) on {'cuda' if device == 0 else 'cpu'}...")
    sam = SAM(weights)
    print("SAM2 loaded.")
    while True:
        frame, frame_id, tracked_objs = sam_in_q.get()
        if frame is None:
            break
        h, w = frame.shape[:2]
        masks, feedback = [], {}
        visible = [o for o in tracked_objs if o['last_seen'] == 0]

        if (SAM_AUTO_DISCOVERY_ENABLED and frame_id > 0
                and frame_id % SAM_AUTOMATIC_EVERY_N_FRAMES == 0):
            res = sam(frame, verbose=False, device=device)
            if res and res[0].masks is not None:
                for seg in res[0].masks.data.cpu().numpy().astype(bool):
                    masks.append({'segmentation': resize_mask(seg, w, h), 'color': DISCOVERED_COLOR})
        elif visible:
            boxes = [list(clamp_bbox(o['bbox'], w, h)) for o in visible]
            res = sam(frame, bboxes=boxes, verbose=False, device=device)
            if res and res[0].masks is not None:
                mdata = res[0].masks.data.cpu().numpy().astype(bool)
                for i, obj in enumerate(visible):
                    if i >= len(mdata):
                        break
                    seg = resize_mask(mdata[i], w, h)
                    rx1, ry1, rx2, ry2 = boxes[i]  # crop to the prompt box -> anchored to the box
                    mask_sub = seg[ry1:ry2, rx1:rx2].copy()
                    m = cv2.moments(seg.astype(np.uint8))
                    contour_pts, cx, cy = [], None, None
                    if m["m00"] > 0:
                        cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                        contours, _ = cv2.findContours(seg.astype(np.uint8),
                                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            eps = CONTOUR_SIMPLIFY_EPS * cv2.arcLength(c, True)
                            contour_pts = cv2.approxPolyDP(c, eps, True).squeeze().tolist()
                    # The mask is delivered relative to the tracker box; the main loop re-warps
                    # it onto each frame's Kalman-smoothed box so it moves smoothly and doesn't jitter.
                    feedback[obj['id']] = {'refined_center': (cx, cy) if cx is not None else None,
                                           'mask_area': int(m["m00"]), 'mask_contour': contour_pts,
                                           'mask_sub': mask_sub, 'mask_ref_bbox': tuple(boxes[i])}
        if sam_masks_q.empty():
            sam_masks_q.put(masks)
        if feedback and sam_feedback_q.empty():
            sam_feedback_q.put(feedback)


# =============================================================================
# Depth thread (Depth Anything V2, metric) — feeds occlusion ordering
# =============================================================================
def depth_worker(model_name):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading depth model ({model_name}) on {device}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
    print("Depth model loaded.")
    while True:
        frame, frame_id = depth_in_q.get()
        if frame is None:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (DEPTH_INFER_SIZE, DEPTH_INFER_SIZE), interpolation=cv2.INTER_AREA)
        inputs = processor(images=small, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        depth = out.predicted_depth.squeeze().float().cpu().numpy()
        depth_map = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        if depth_out_q.empty():
            depth_out_q.put((frame_id, depth_map))


# =============================================================================
# Rendering
# =============================================================================
def blend_mask(display, seg, color, alpha=0.5):
    if seg is not None and seg.any():
        col = np.array(color, np.float32)
        display[seg] = (display[seg] * (1 - alpha) + col * alpha).astype(np.uint8)


def place_mask(mask_sub, cur_bbox, w, h):
    """Resize a box-cropped SAM mask onto the current Kalman box -> mask follows the smooth
    box (no jitter) and tracks the object between the slower SAM updates."""
    if mask_sub is None or mask_sub.size == 0:
        return None
    x1, y1, x2, y2 = clamp_bbox(cur_bbox, w, h)
    cw, ch = x2 - x1, y2 - y1
    if cw < 1 or ch < 1:
        return None
    resized = cv2.resize(mask_sub.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = np.zeros((h, w), bool)
    out[y1:y2, x1:x2] = resized
    return out


def draw_dashed_rect(img, p1, p2, color, thickness=2, dash=8):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_overlays(display, tracks, fps):
    for t in tracks:
        color = COLORS[abs(t['id']) % len(COLORS)]
        x1, y1, x2, y2 = t['bbox']
        center = tuple(map(int, t.get('refined_center') or t['bbox_center']))
        status = t['status']
        if status == 'TRACKED':
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        else:  # PREDICTED / COASTING / OCCLUDED / DORMANT: dashed + dimmed
            dim = tuple(int(c * 0.6) for c in color)
            draw_dashed_rect(display, (x1, y1), (x2, y2), dim, 2)
        cv2.circle(display, center, 5, color, -1)

        tag = '' if status == 'TRACKED' else f" ({status.lower()})"
        cv2.putText(display, f"ID {t['id']} {t['class_name']}{tag}", (x1, y1 - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(display,
                    f"{t['motion_state']} {t.get('speed_norm', 0.0):.2f}/s {t.get('direction', '-')} "
                    f"c={t.get('track_confidence', 0.0):.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if t['motion_state'] == 'moving':
            vx, vy = t.get('velocity', (0.0, 0.0))
            end = (int(center[0] + vx * MOTION_ARROW_LOOKAHEAD_S),
                   int(center[1] + vy * MOTION_ARROW_LOOKAHEAD_S))
            cv2.arrowedLine(display, center, end, color, 2, tipLength=0.3)

    cv2.putText(display, f"FPS {fps:4.1f}  tracks {len(tracks)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def build_records(tracks, w, h, now):
    out = []
    for t in tracks:
        bbox = t.get('bbox')
        bbox_norm = ([safe_num(bbox[0] / w), safe_num(bbox[1] / h),
                      safe_num(bbox[2] / w), safe_num(bbox[3] / h)] if bbox else None)
        first_seen_t = t.get('first_seen_t')
        out.append({
            "object_id": t.get('id'),
            "tracker_id": t.get('tracker_id'),
            "class_name": t.get('class_name'),
            "status": t.get('status'),
            "occluded_by": t.get('occluded_by'),
            "track_confidence": safe_num(t.get('track_confidence', 0.0), 3),
            "max_confidence": safe_num(t.get('max_confidence', 0.0), 3),
            "pixel_locked": t.get('pixel_locked', False),
            "detection_confidence": safe_num(t.get('confidence', 0.0), 4),
            "sam_processed": 'mask_area' in t,
            "bbox": bbox,
            "bbox_norm": bbox_norm,
            "bbox_center": t.get('bbox_center'),
            "refined_center": t.get('refined_center'),
            "mask_area": t.get('mask_area'),
            "depth_m": safe_num(t.get('depth'), 3),
            "velocity_px_s": [safe_num(v, 2) for v in t.get('velocity', (0.0, 0.0))],
            "speed_px_s": safe_num(t.get('speed_px_s', 0.0), 2),
            "speed_norm": safe_num(t.get('speed_norm', 0.0), 4),
            "direction": t.get('direction'),
            "direction_deg": safe_num(t.get('direction_deg', 0.0), 1),
            "motion_state": t.get('motion_state'),
            "trajectory": list(t.get('trajectory', [])),
            "mask_contour": t.get('mask_contour'),
            "age_s": safe_num(now - first_seen_t, 3) if first_seen_t is not None else None,
            "detections_seen": t.get('detections', 0),
            "since_last_detection_s": safe_num(now - t.get('last_detection_t', now), 3),
        })
    return out


# =============================================================================
# Main = fast tracker (every frame): optical flow + Kalman, corrected by detections
# =============================================================================
def main():
    import torch
    # BoT-SORT appearance ReID is ~100x slower on CPU (seconds/frame); only enable on GPU.
    reid = BOTSORT_WITH_REID and torch.cuda.is_available()
    if BOTSORT_WITH_REID and not reid:
        print("Note: no CUDA -> BoT-SORT ReID disabled (motion-only association).")
    write_tracker_cfg(TRACKER_RUNTIME_CFG, reid)

    yolo_model = YOLO(YOLO_WEIGHTS)
    det_thread = threading.Thread(target=detection_worker, args=(yolo_model,), daemon=True)
    det_thread.start()

    sam_running = SAM_ENABLED and torch.cuda.is_available()
    sam_thread = None
    if sam_running:
        sam_thread = threading.Thread(target=sam_worker, args=(SAM2_WEIGHTS,), daemon=True)
        sam_thread.start()
    elif SAM_ENABLED:
        print("\nWARNING: No CUDA GPU detected. SAM2 disabled (too slow on CPU).\n")

    depth_running = DEPTH_ENABLED and torch.cuda.is_available()
    depth_thread = None
    if depth_running:
        depth_thread = threading.Thread(target=depth_worker, args=(DEPTH_MODEL,), daemon=True)
        depth_thread.start()
    elif DEPTH_ENABLED:
        print("\nWARNING: No CUDA GPU detected. Depth disabled -> occlusion falls back to "
              "IOU-only heuristic (can't tell which overlapping object is in front).\n")

    cap = cv2.VideoCapture(SOURCE)
    if isinstance(SOURCE, int):
        if CAMERA_WIDTH:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        if CAMERA_HEIGHT: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if CAMERA_FPS:    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    if not cap.isOpened():
        print(f"Error: could not open source {SOURCE!r}.")
        return

    tracks = {}          # object_id -> track dict (active)
    lost_memory = {}     # object_id -> {bbox, velocity, class_name, hist, lost_t, kf}
    tid2oid = {}         # BoT-SORT tracker_id -> object_id
    next_object_id = 0

    writer = None
    log_file = open(LOG_PATH, 'w') if LOG_ENABLED else None
    latest_masks = []
    latest_depth = None      # (h,w) float32 metric depth map, or None if depth is disabled/cold
    prev_gray = None
    fps, last_t = 0.0, time.time()
    frame_id = 0

    def make_track(oid, tid, kf, bbox, conf, cls, now, gray, frame, velocity=(0.0, 0.0),
                   first_seen_t=None, prior_detections=0, prior_max_conf=0.0):
        sbbox, scenter, vel = kf.state()
        det_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        t = {
            'id': oid, 'tracker_id': tid, 'kf': kf,
            'feat_pts': seed_points(gray, bbox), 'hist': color_hist(frame, bbox),
            'bbox': sbbox, 'bbox_center': scenter, 'class_name': cls,
            'confidence': conf, 'track_confidence': conf, 'status': 'TRACKED',
            'pixel_locked': True, 'last_detection_t': now, 'last_detection_center': det_center,
            'velocity': velocity, 'speed_px_s': 0.0, 'speed_norm': 0.0,
            'direction': '-', 'direction_deg': 0.0, 'motion_state': 'stationary',
            'trajectory': deque([scenter], maxlen=TRAJECTORY_LEN),
            # Persisted across re-acquisition (see pick_up) so an object's history in the log
            # doesn't reset to "just seen once" every time it's re-associated after a gap.
            'first_seen_t': first_seen_t if first_seen_t is not None else now,
            'max_confidence': max(prior_max_conf, conf), 'detections': prior_detections + 1,
            'settled': False, 'dormant_missing': 0,
        }
        apply_motion(t, sbbox, scenter, vel, diag)
        return t

    def new_track(tid, bbox, conf, cls, now, gray, frame):
        nonlocal next_object_id
        oid = next_object_id
        next_object_id += 1
        kf = KalmanBoxTracker(bbox, now)
        tracks[oid] = make_track(oid, tid, kf, bbox, conf, cls, now, gray, frame)
        tid2oid[tid] = oid
        return oid

    def pick_up(tid, bbox, conf, cls, now, gray, frame, matched):
        """Broad re-acquisition. Search a radius that GROWS with the candidate's speed and
        time-since-lost, over dormant/coasting tracks AND recent lost memory, gated by class
        + appearance. This is how fast movers and stationary occluded objects get picked back up."""
        frame_hist = color_hist(frame, bbox)
        frame_depth = track_depth(latest_depth, bbox) if latest_depth is not None else None
        dcx, dcy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        r_base = REID_BASE_RADIUS_NORM * diag
        best = {'oid': None, 'src': None, 'score': -1e9}

        def consider(oid, center, velocity, cls_name, hist, ref_t, source, depth):
            if cls_name != cls:
                return
            dt_lost = max(now - ref_t, 0.0)
            vx, vy = velocity
            pcx, pcy = center[0] + vx * dt_lost, center[1] + vy * dt_lost
            d = math.hypot(dcx - pcx, dcy - pcy)
            radius = r_base + math.hypot(vx, vy) * dt_lost
            if d > radius:
                return
            sim = hist_sim(frame_hist, hist)
            if sim < REID_HIST_SIM_BROAD:
                return
            # Depth is a bonus disambiguator, not a requirement: two same-class, similarly-
            # colored objects at very different distances from the camera shouldn't be
            # confused for each other. Skipped entirely when either side's depth is unknown.
            if frame_depth is not None and depth is not None and abs(frame_depth - depth) > REID_DEPTH_TOL_M:
                return
            score = sim - d / max(radius, 1.0)
            if score > best['score']:
                best.update(oid=oid, src=source, score=score)

        for oid, t in tracks.items():
            if oid in matched or t['status'] == 'TRACKED':
                continue
            consider(oid, t['bbox_center'], t['velocity'], t['class_name'], t['hist'],
                     t.get('last_detection_t', now), 'active', t.get('depth'))
        for oid, mem in lost_memory.items():
            consider(oid, mem['center'], mem['velocity'], mem['class_name'], mem['hist'],
                     mem['lost_t'], 'lost', mem.get('depth'))

        if best['oid'] is None:
            return None
        oid = best['oid']
        # Carry the prior identity's history forward (first-seen time, total detections, peak
        # confidence) instead of restarting it -> the log's persistence stats stay accurate
        # across an occlusion/loss gap instead of resetting every re-acquisition.
        if best['src'] == 'lost':
            prior = lost_memory.pop(oid)
        else:
            prior = tracks[oid]
        kf = prior['kf']
        kf.predict(now)
        kf.correct(bbox, r_scale=1.0)
        _, _, vel = kf.state()
        tracks[oid] = make_track(oid, tid, kf, bbox, conf, cls, now, gray, frame, velocity=vel,
                                  first_seen_t=prior.get('first_seen_t'),
                                  prior_detections=prior.get('detections', 0),
                                  prior_max_conf=prior.get('max_confidence', 0.0))
        tid2oid[tid] = oid
        return oid

    while True:
        success, frame = cap.read()
        if not success:
            break
        now = time.time()
        dt = now - last_t
        last_t = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

        frame = apply_clahe(frame)
        display = frame.copy()
        h, w = frame.shape[:2]
        diag = math.hypot(w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 1) Fast tracker: advance every track with optical flow + Kalman ---
        for t in tracks.values():
            if t['status'] in ('DORMANT', 'OCCLUDED'):
                # Frozen in place: an OCCLUDED track's box is hidden behind a nearer object, so
                # neither optical flow (which would just pick up the occluder's motion) nor
                # Kalman predict runs -> it holds its last known position instead of drifting or
                # getting dragged along by whatever is in front of it. Detections in step 2 still
                # correct it directly and un-freeze it (status re-evaluated every frame in step 4).
                continue
            prev_bbox = t['bbox']
            t['kf'].predict(now)
            ok = False
            if prev_gray is not None and t['feat_pts'] is not None:
                new_pts, mdx, mdy, flow_ok = flow_step(prev_gray, gray, t['feat_pts'])
                if flow_ok:
                    flow_bbox = (prev_bbox[0] + mdx, prev_bbox[1] + mdy,
                                 prev_bbox[2] + mdx, prev_bbox[3] + mdy)
                    # An object's flow is easily hijacked by someone passing in front of it.
                    # If the shifted region no longer looks like the object, OR it moves into
                    # another tracked object's space, OR the depth under it suddenly doesn't
                    # match the object's own depth (an occluder physically sits at a different
                    # distance from the camera), it's an occluder -> ignore the flow. Depth is
                    # checked regardless of 'settled' -> unlike appearance/IOU, it also catches
                    # a MOVING track's flow getting hijacked, which previously had no protection.
                    trust = True
                    depth_jump = False
                    if latest_depth is not None and t.get('depth') is not None:
                        flow_depth = track_depth(latest_depth, clamp_bbox(flow_bbox, w, h))
                        depth_jump = (flow_depth is not None and
                                      abs(flow_depth - t['depth']) > DEPTH_HIJACK_DELTA_M)
                    if depth_jump or (t.get('settled') and math.hypot(mdx, mdy) > DORMANT_WAKE_PX):
                        is_hijacked = False
                        for other_t in tracks.values():
                            if other_t['id'] == t['id'] or other_t['status'] != 'TRACKED':
                                continue
                            if calculate_iou(flow_bbox, other_t['bbox']) > FLOW_MATCH_SIM:
                                is_hijacked = True
                                break
                        appearance_mismatch = hist_sim(
                            color_hist(frame, clamp_bbox(flow_bbox, w, h)), t['hist']) < FLOW_MATCH_SIM
                        if is_hijacked or appearance_mismatch or depth_jump:
                            trust = False
                            ok = t['pixel_locked'] = False
                    if trust:
                        t['feat_pts'] = new_pts
                        t['kf'].correct(flow_bbox, r_scale=OPTFLOW_MEASUREMENT_R_SCALE)
                        ok = True
            t['pixel_locked'] = ok

            sbbox, scenter, vel = t['kf'].state()
            sbbox = clamp_bbox(sbbox, w, h)          # never let a phantom box leave the screen
            scenter = ((sbbox[0] + sbbox[2]) // 2, (sbbox[1] + sbbox[3]) // 2)
            apply_motion(t, sbbox, scenter, vel, diag)
            t['trajectory'].append(scenter)
            # If flow didn't lock, (re)acquire points on the predicted box so the next
            # frame can re-lock onto the object mid-gap instead of coasting blindly.
            if not ok:
                t['feat_pts'] = seed_points(gray, sbbox)

        # --- 1b) Maintain dormant objects: wake only if THEY move (not an occluder), and
        #         distinguish transient occlusion (keep) from removal (retire). ---
        for oid in list(tracks.keys()):
            t = tracks[oid]
            if t['status'] != 'DORMANT':
                continue
            box = t['bbox']
            present_here = hist_sim(color_hist(frame, box), t['hist']) >= DORMANT_PRESENCE_SIM
            moved, mdx, mdy, new_pts = False, 0.0, 0.0, None
            if prev_gray is not None and t.get('feat_pts') is not None:
                np_, dx, dy, ok = flow_step(prev_gray, gray, t['feat_pts'])
                if ok and math.hypot(dx, dy) > DORMANT_WAKE_PX:
                    shifted = clamp_bbox((box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy), w, h)
                    # Wake only if the object ITSELF moved: its texture must still match at the new
                    # spot. A person walking in front hijacks the flow but won't match appearance.
                    if hist_sim(color_hist(frame, shifted), t['hist']) >= DORMANT_PRESENCE_SIM:
                        moved, mdx, mdy, new_pts = True, dx, dy, np_

            if moved:
                prev_bbox = t['bbox']
                t['kf'].predict(now)
                t['kf'].correct((prev_bbox[0] + mdx, prev_bbox[1] + mdy,
                                 prev_bbox[2] + mdx, prev_bbox[3] + mdy),
                                r_scale=OPTFLOW_MEASUREMENT_R_SCALE)
                t['kf'].set_velocity(mdx / dt if dt > 0 else 0.0, mdy / dt if dt > 0 else 0.0, 0.7)
                sbbox, scenter, vel = t['kf'].state()
                sbbox = clamp_bbox(sbbox, w, h)
                apply_motion(t, sbbox, scenter, vel, diag)
                t['feat_pts'] = new_pts
                t.update({'status': 'PREDICTED', 'pixel_locked': True,
                          'last_detection_t': now, 'settled': False, 'dormant_missing': 0})
            elif present_here:
                # Still sitting there (or reappeared after an occluder passed).
                t['dormant_missing'] = 0
                t['feat_pts'] = seed_points(gray, box)   # keep points on the visible object
            else:
                # Occluded (transient) or removed (sustained): hold position, count toward retire.
                t['dormant_missing'] = t.get('dormant_missing', 0) + 1
                if t['dormant_missing'] > DORMANT_MISSING_MAX:
                    tid2oid.pop(t['tracker_id'], None)
                    del tracks[oid]

        # --- 2) Fuse detections: correct known ids, broadly re-acquire the rest ---
        try:
            _fid, _fts, dets = det_out_q.get_nowait()
            matched = set()
            # Pass 1: detections whose tracker_id we already own -> strong correction,
            # and inject the directly-measured velocity so speed doesn't lag.
            for tid, bbox, conf, cls in dets:
                oid = tid2oid.get(tid)
                if oid is None or oid not in tracks:
                    continue
                t = tracks[oid]
                det_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                t['kf'].correct(bbox, r_scale=1.0)
                dtl = now - t['last_detection_t']
                if dtl > 1e-3 and t.get('last_detection_center'):
                    mvx = (det_center[0] - t['last_detection_center'][0]) / dtl
                    mvy = (det_center[1] - t['last_detection_center'][1]) / dtl
                    t['kf'].set_velocity(mvx, mvy, VELOCITY_INJECT)
                sbbox, scenter, vel = t['kf'].state()
                apply_motion(t, sbbox, scenter, vel, diag)
                t.update({'feat_pts': seed_points(gray, bbox), 'hist': color_hist(frame, bbox),
                          'confidence': conf, 'track_confidence': conf, 'class_name': cls,
                          'last_detection_t': now, 'last_detection_center': det_center,
                          'status': 'TRACKED', 'pixel_locked': True,
                          'max_confidence': max(t.get('max_confidence', 0.0), conf),
                          'detections': t.get('detections', 0) + 1,
                          'settled': t['speed_norm'] < MOTION_SPEED_THRESHOLD_NORM})
                matched.add(oid)
            # Pass 2: unknown tracker_ids -> broad pickup (revives dormant/lost), else new.
            for tid, bbox, conf, cls in dets:
                if tid2oid.get(tid) in matched:
                    continue
                oid = pick_up(tid, bbox, conf, cls, now, gray, frame, matched)
                if oid is None:
                    oid = new_track(tid, bbox, conf, cls, now, gray, frame)
                matched.add(oid)
        except queue.Empty:
            pass

        # --- 3) Merge SAM feedback (by object_id) ---
        try:
            for oid, refined in sam_feedback_q.get_nowait().items():
                if oid in tracks:
                    tracks[oid].update(refined)
        except queue.Empty:
            pass

        # --- 3b) Sample per-track depth from the latest depth map (kept stale if depth is
        #         disabled/cold, so occlusion logic below always has last-known-good data). ---
        if latest_depth is not None:
            for t in tracks.values():
                d = track_depth(latest_depth, t['bbox'])
                if d is not None:
                    t['depth'] = d

        # --- 4) Confidence decay, status, dormancy, retirement ---
        for oid in list(tracks.keys()):
            t = tracks[oid]
            if t['status'] == 'DORMANT':
                if DORMANT_MAX_S and now - t.get('dormant_t', now) > DORMANT_MAX_S:
                    lost_memory[oid] = {'bbox': t['bbox'], 'center': t['bbox_center'],
                                        'velocity': t['velocity'], 'class_name': t['class_name'],
                                        'hist': t['hist'], 'lost_t': now, 'kf': t['kf'],
                                        'depth': t.get('depth'),
                                        'first_seen_t': t.get('first_seen_t'),
                                        'detections': t.get('detections', 0),
                                        'max_confidence': t.get('max_confidence', 0.0)}
                    tid2oid.pop(t['tracker_id'], None)
                    del tracks[oid]
                continue

            since_det = now - t['last_detection_t']

            # Occlusion: an overlapping box is only "occluding" if it's the one in front.
            # With depth available, the farther-from-camera track is occluded and the nearer
            # one stays TRACKED/unaffected. Without depth (no CUDA / cold start), fall back
            # to the old symmetric IOU-only guess so behavior on the CPU dev box is unchanged.
            is_occluded = False
            occluder_id = None
            if since_det > 0.05:  # Only check for occlusion if not actively detected
                for other_t in tracks.values():
                    if other_t['id'] == t['id'] or other_t['status'] != 'TRACKED':
                        continue
                    if calculate_iou(t['bbox'], other_t['bbox']) <= OCCLUSION_IOU_THRESHOLD:
                        continue
                    t_depth, o_depth = t.get('depth'), other_t.get('depth')
                    if t_depth is not None and o_depth is not None:
                        if t_depth > o_depth:      # t is farther away -> t is behind -> occluded
                            is_occluded, occluder_id = True, other_t['id']
                            break
                    else:
                        is_occluded, occluder_id = True, other_t['id']  # no depth yet: old fallback
                        break

            # Record who's occluding whom -> a downstream algorithm/LLM reading the log can
            # explain *why* an object is hidden instead of just seeing a bare "OCCLUDED" status.
            t['occluded_by'] = occluder_id if is_occluded else None

            if is_occluded:
                t['status'] = 'OCCLUDED'
                # Don't decay confidence while occluded, it's just hidden.
                # Refresh confidence to a high baseline so it can survive a long occlusion.
                t['track_confidence'] = max(t['track_confidence'], t.get('max_confidence', 0.8) * 0.9)
            elif since_det > TRACKED_HOLD_S:
                t['track_confidence'] *= math.exp(-CONF_DECAY_PER_S * dt)
                t['status'] = 'PREDICTED' if t['pixel_locked'] else 'COASTING'
            elif t['status'] != 'TRACKED':
                 t['status'] = 'TRACKED'

            # Retire if confidence is too low (and not occluded) or lost for too long
            if t['status'] != 'OCCLUDED' and (since_det > COAST_MAX_S or t['track_confidence'] < CONF_FLOOR):
                lost_memory[oid] = {'bbox': t['bbox'], 'center': t['bbox_center'],
                                    'velocity': t['velocity'], 'class_name': t['class_name'],
                                    'hist': t['hist'], 'lost_t': now, 'kf': t['kf'],
                                    'depth': t.get('depth'),
                                    'first_seen_t': t.get('first_seen_t'),
                                    'detections': t.get('detections', 0),
                                    'max_confidence': t.get('max_confidence', 0.0)}
                tid2oid.pop(t['tracker_id'], None)
                del tracks[oid]

        for oid in [o for o, m in lost_memory.items() if now - m['lost_t'] > REID_MEMORY_S]:
            del lost_memory[oid]

        # --- 5) Feed detection + SAM + depth threads, draw, log ---
        if det_in_q.empty():
            det_in_q.put((frame, frame_id, now))
        active = list(tracks.values())
        if sam_running and sam_in_q.empty():
            sam_in_q.put((frame, frame_id, [{'id': t['id'], 'bbox': t['bbox'], 'last_seen': 0}
                                            for t in active if t['status'] in ('TRACKED', 'PREDICTED')]))
        if sam_running and not sam_masks_q.empty():
            latest_masks = sam_masks_q.get_nowait()
        if depth_running and frame_id % DEPTH_EVERY_N_FRAMES == 0 and depth_in_q.empty():
            depth_in_q.put((frame, frame_id))
        if depth_running and not depth_out_q.empty():
            _dfid, latest_depth = depth_out_q.get_nowait()

        # Discovered (untracked) masks are drawn where SAM found them.
        for m in latest_masks:
            blend_mask(display, m.get('segmentation'), m.get('color', DISCOVERED_COLOR))
        # Tracked-object masks are anchored to each track's current Kalman box every frame.
        for t in active:
            warped = place_mask(t.get('mask_sub'), t['bbox'], w, h)
            blend_mask(display, warped, COLORS[abs(t['id']) % len(COLORS)])
        draw_overlays(display, active, fps)

        records = build_records(active, w, h, now)
        if LOG_ENABLED and records:
            log_file.write(json.dumps(
                {"schema_version": LOG_SCHEMA_VERSION, "frame_id": frame_id,
                 "timestamp": round(now, 4), "timestamp_iso": datetime.fromtimestamp(now).isoformat(),
                 "fps": safe_num(fps, 1), "frame_width": w, "frame_height": h,
                 "objects": records}, cls=NumpyEncoder) + "\n")
            log_file.flush()
        if PRINT_SUMMARY and records:
            print(f"[f{frame_id} {fps:4.1f}fps] " + ", ".join(
                f"#{r['object_id']}:{r['class_name']}[{r['status']} {r['motion_state']} "
                f"{r['speed_norm'] or 0.0:.2f} c{r['track_confidence'] or 0.0:.2f}]" for r in records))

        if SAVE_VIDEO_ENABLED:
            if writer is None:
                writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH,
                                         cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
            writer.write(display)

        if not HEADLESS_ENABLED:
            cv2.imshow('YOLO + BoT-SORT + optical-flow + Kalman', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        prev_gray = gray
        frame_id += 1

    det_in_q.put((None, -1, 0))
    if sam_running:
        sam_in_q.put((None, -1, None))
    if depth_running:
        depth_in_q.put((None, -1))
    det_thread.join(timeout=3)
    if sam_thread is not None:
        sam_thread.join(timeout=3)
    if depth_thread is not None:
        depth_thread.join(timeout=3)
    if writer:
        writer.release()
    if log_file:
        log_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
