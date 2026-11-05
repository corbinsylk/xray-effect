#!/usr/bin/env python3
"""The x-ray flicker on your own talking head clip.

Ten flashes in 0.8 seconds, each a new look on the person only: an x-ray, a cyan negative or a white line drawing, on
black or your colour. The room never changes. The camera accelerates in and lands right after the last flash, holds, then
eases back. Everything runs on this computer: ready-made x-ray parts (parts/) are fitted to the body by MediaPipe's
free tracking on every flash, so the skeleton stays on the pose and no image model or API key is needed.

Run it through run.py, which sets everything up first:  python3 run.py my-clip.mp4 --start 0
"""
import argparse, re, subprocess, sys
from pathlib import Path
from fractions import Fraction
import cv2, numpy as np, imageio_ffmpeg, mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision as V

# =========================== THE LOOK (defaults are the approved look; times are seconds) ============================
RUN_FPS = 24                # the flicker is timed on a 24 fps grid at every frame rate
FIRST_FLASH = 2             # the first flash lands on grid frame 2 after --start
FLASH_STEP = 2              # then one flash on every second grid frame: as fast as a flicker can go
LOOKS = ["xray", "neg", "wire", "xray_colour", "neg", "wire_colour", "xray", "neg", "wire", "xray_colour"]
LAND = 21                   # the camera lands on grid frame 21
PUNCH = 1.5                 # how far the camera goes in
ACCEL = 2.6                 # the push accelerates: progress ** 2.6, in log space, so it hits hard at the end
HOLD = 0.95                 # seconds held after landing, creeping 3 % so the frame is never dead
CREEP = 1.03
RETURN = 0.9                # seconds to ease back to the whole frame
OUT_CURVE = [0, 0.14, 0.37, 0.57, 0.71, 0.8, 0.87, 0.94, 1]   # the measured ease of the camera's return
HEADROOM = 0.052            # the top of the head never goes above this share of the frame height
WORK_SIZE = 1920            # the looks are built at up to this long side, then scaled to the clip
# ======================================================================================================================

HERE = Path(__file__).resolve().parent
PARTS, MODELS = HERE / "parts", HERE / "models"


def die(msg):
    print(f"\nX-RAY STOPPED: {msg}\n", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------ the clip ------------------------------------------------------
FF = imageio_ffmpeg.get_ffmpeg_exe()
NICE = {23.98: "24000/1001", 23.976: "24000/1001", 29.97: "30000/1001", 59.94: "60000/1001", 119.88: "120000/1001"}


def probe(clip):
    info = subprocess.run([FF, "-hide_banner", "-i", str(clip)], capture_output=True, text=True, errors="replace").stderr
    video = next((l for l in info.splitlines() if "Video:" in l), None)
    if not video:
        die(f"could not read a video from {clip}")
    m = re.search(r"([\d.]+) fps", video) or re.search(r"([\d.]+) tbr", video)
    fps = float(m.group(1)) if m else 30.0
    rate = NICE.get(fps) or str(Fraction(fps).limit_denominator(1001))
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
    seconds = int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)) if dur else None
    hdr = any(k in video for k in ("arib-std-b67", "smpte2084"))
    return float(Fraction(rate)), rate, seconds, hdr


def decode_filter(rate, hdr):
    """fps first, so a phone's variable frame rate cannot shift the timing. HDR phone footage is mapped to normal video."""
    chain = [f"fps={rate}"]
    if hdr:
        filters = subprocess.run([FF, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        if " zscale " in filters and " tonemap " in filters:
            chain.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv")
        else:
            print("This clip is HDR and this ffmpeg cannot convert it, so its colours may look flat. Export it as normal (SDR) video if so.")
    return ",".join(chain + ["format=bgr24"])


def frames(clip, vf, size):
    w, h = size
    p = subprocess.Popen([FF, "-v", "error", "-nostdin", "-i", str(clip), "-vf", vf, "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    try:
        while True:
            b = p.stdout.read(w * h * 3)
            if len(b) < w * h * 3:
                return
            yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def clip_size(clip, vf):
    """from a real decoded frame, so a phone's rotation flag is already applied"""
    r = subprocess.run([FF, "-v", "error", "-nostdin", "-i", str(clip), "-vf", vf.replace("format=bgr24", "format=gray"), "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"], capture_output=True)
    im = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_GRAYSCALE) if r.stdout else None
    if im is None:
        die(f"could not decode {clip}")
    return im.shape[1], im.shape[0]


# ---------------------------------------------------- the tracking ----------------------------------------------------
def tracker():
    opt = lambda name: BaseOptions(model_asset_path=str(MODELS / name))
    return dict(
        pose=V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(base_options=opt("pose_landmarker_heavy.task"))),
        face=V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(base_options=opt("face_landmarker.task"), min_face_detection_confidence=0.2, min_face_presence_confidence=0.2)),
        hand=V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(base_options=opt("hand_landmarker.task"), num_hands=2, min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2)),
        seg=V.ImageSegmenter.create_from_options(V.ImageSegmenterOptions(base_options=opt("selfie_multiclass_256x256.tflite"), output_confidence_masks=True)),
    )


def lift_of(frame):
    """gamma 0.45, so dark hair, a dark cap and the shadow side separate from a dark room"""
    return (255 * (frame / 255.0) ** 0.45).clip(0, 255).astype(np.uint8)


def mp_image(bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


POSE_POINTS = [11, 12, 13, 14, 15, 16, 23, 24]


def track(T, frame, lift):
    im, im0 = mp_image(lift), mp_image(frame)
    pr, fr, hr = T["pose"].detect(im), T["face"].detect(im), T["hand"].detect(im)
    if not fr.face_landmarks:
        fr = T["face"].detect(im0)
    hands = [[(p.x, p.y) for p in h] for h in hr.hand_landmarks] + [[(p.x, p.y) for p in h] for h in T["hand"].detect(im0).hand_landmarks]
    return {
        "pose": {i: (pr.pose_landmarks[0][i].x, pr.pose_landmarks[0][i].y) for i in POSE_POINTS} if pr.pose_landmarks else None,
        "face": [(p.x, p.y) for p in fr.face_landmarks[0]] if fr.face_landmarks else None,
        "hands": hands,
    }


def nearest(TR, n, get, span=6):
    for off in sorted(range(-span, span + 1), key=abs):
        v = get(TR.get(n + off) or {})
        if v:
            return v


def hands_at(TR, n, aspect):
    """one believable hand per wrist, from this frame or the two either side. Blurred or foreshortened hands are left out, never guessed."""
    P = nearest(TR, n, lambda t: t.get("pose"))
    sw = np.hypot((P[11][0] - P[12][0]) * aspect, P[11][1] - P[12][1])
    out = []
    for w in (15, 16):
        def near(t):
            ok = []
            for h in t.get("hands", []):
                pts = np.array(h)
                palm = np.hypot((pts[0][0] - pts[9][0]) * aspect, pts[0][1] - pts[9][1])
                wd = np.hypot((pts[0][0] - P[w][0]) * aspect, pts[0][1] - P[w][1])
                tips = np.hypot((pts[[8, 12, 16, 20], 0] - pts[0][0]) * aspect, pts[[8, 12, 16, 20], 1] - pts[0][1])
                inside = ((pts >= 0.0) & (pts <= 1.0)).all(1).mean()
                if inside > 0.9 and wd < sw * 0.35 and 0.18 * sw < palm < 0.6 * sw and tips.max() < palm * 2.4:
                    ok.append(h)
            return min(ok, key=lambda h: np.hypot((h[0][0] - P[w][0]) * aspect, h[0][1] - P[w][1])) if ok else None
        h = nearest(TR, n, near, span=2)
        if h:
            out.append(h)
    return out


def person_mask(T, lift):
    cm = T["seg"].segment(mp_image(lift)).confidence_masks
    p = 1 - cm[0].numpy_view().squeeze()      # everything that is not background: hair, skin, clothes, accessories
    p = cv2.resize(p.astype(np.float32), lift.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    guide = cv2.GaussianBlur(lift, (0, 0), 1.2)
    p = cv2.ximgproc.guidedFilter(guide, p, 24, 4e-3).clip(0, 1)
    p = ((p - 0.42) / 0.16).clip(0, 1)
    p = cv2.GaussianBlur(p, (0, 0), 1.2)
    n, lab, st, _ = cv2.connectedComponentsWithStats((p > 0.5).astype(np.uint8))
    if n > 1:
        keep = cv2.dilate((lab == 1 + st[1:, cv2.CC_STAT_AREA].argmax()).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        p = p * keep
    return p


# ---------------------------------------------------- the x-ray parts ---------------------------------------------------
def sprite(name):
    g = cv2.imread(str(PARTS / f"{name}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    g = ((g - 0.07) / 0.93).clip(0, 1)
    ys = np.arange(g.shape[0], dtype=np.float32)
    if name == "torso":      # both ends fade: the part stops at the image's edges, which would read as hard lines at the chin and the pelvis
        g = g * (np.clip((ys - 40) / 200, 0, 1) * np.clip((g.shape[0] - 30 - ys) / 170, 0, 1))[:, None]
    if name == "skull":      # lighter above the brow, so a cap or hair stays readable over the cranium
        g = g * (0.45 + 0.55 * np.clip((ys - 200) / 170, 0, 1))[:, None]
    if name == "forearm":    # the haze beside the bones
        g = ((g - 0.2) / 0.8).clip(0, 1) ** 1.2
    return g


def limb_box(g, t=0.3):
    ys = np.where((g > t).any(1))[0]
    xs = np.where((g > t).mean(0) > 0.03)[0]
    return (xs[0] + xs[-1]) / 2, ys[0], ys[-1], xs[-1] - xs[0]


def load_parts():
    spr = {n: sprite(n) for n in ("skull", "torso", "humerus", "forearm", "phalanx")}
    return spr, {n: limb_box(spr[n]) for n in ("humerus", "forearm", "phalanx")}


def lay(c, g, src, dst):
    M = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    w = cv2.warpAffine(g, M, c.shape[1::-1], flags=cv2.INTER_CUBIC, borderValue=0).clip(0, 1)
    c[:] = 1 - (1 - c) * (1 - w)


def limb(c, SPR, BOX, name, a, b, width, over=0.06):
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    L = np.hypot(*d)
    if L < 3:
        return
    a2, b2 = a - d * over, b + d * over
    p = np.array([-d[1], d[0]]) / L
    cx, y0, y1, bw = BOX[name]
    lay(c, SPR[name], [(cx, y0), (cx, y1), (cx + bw / 2, y0)], [a2, b2, a2 + p * width / 2])


# ------------------------------------------------------ the looks ------------------------------------------------------
def tissue(lift, m):
    """the translucent layer: the frame's own detail with a bright rim from the mask, so the cap, shirt and hair are really theirs"""
    l = cv2.cvtColor(lift, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    dist = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    rim = np.exp(-dist / 16)
    hp = np.abs(l - cv2.GaussianBlur(l, (0, 0), 7)) * 2.6
    hp = cv2.GaussianBlur(hp, (0, 0), 0.8)
    return (0.14 + 0.14 * (1 - l) ** 1.2 * np.exp(-dist / 90) + 0.34 * rim + hp * 0.7).clip(0, 1)


def xray(lift, m, tr, SPR, BOX):
    H, W = m.shape
    c = np.zeros((H, W), np.float32)
    P = {k: (v[0] * W, v[1] * H) for k, v in tr["pose"].items()}
    sw = np.hypot(P[11][0] - P[12][0], P[11][1] - P[12][1])
    # the part's shoulder joints onto the tracked shoulders, its spine onto the hip midpoint (image left = the person's right)
    hip = ((P[23][0] + P[24][0]) / 2, (P[23][1] + P[24][1]) / 2)
    lay(c, SPR["torso"], [(72, 307), (749, 307), (410, 1300)], [P[12], P[11], hip])
    for s, e, wr in ((12, 14, 16), (11, 13, 15)):
        limb(c, SPR, BOX, "humerus", P[s], P[e], sw * 0.13)
        limb(c, SPR, BOX, "forearm", P[e], P[wr], sw * 0.17)
    for h in tr["hands"]:
        Q = [(x * W, y * H) for x, y in h]
        palm = np.hypot(Q[0][0] - Q[9][0], Q[0][1] - Q[9][1])
        bones = [(0, 5, .17), (0, 9, .17), (0, 13, .16), (0, 17, .15), (1, 2, .17), (2, 3, .15), (3, 4, .13)]
        bones += [(j, j + 1, .15 - (j % 4) * .015) for f in (5, 9, 13, 17) for j in range(f, f + 3)]
        for a, b, wd in bones:
            limb(c, SPR, BOX, "phalanx", Q[a], Q[b], palm * wd, over=0.0)
    if tr["face"]:
        F = [(x * W, y * H) for x, y in tr["face"]]
        eye = lambda A, B: ((F[A][0] + F[B][0]) / 2, (F[A][1] + F[B][1]) / 2)
        # the skull's eye sockets and chin onto the tracked eyes and chin
        lay(c, SPR["skull"], [(364, 422), (660, 422), (512, 973)], [eye(33, 133), eye(263, 362), F[152]])
    t = tissue(lift, m)
    return (1 - (1 - t) * (1 - c * 1.05).clip(0, 1)) * m


FACE_MESH = V.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION


def mesh(m, tr):
    """the wireframe: the face's own tracked mesh, and rings round the forearms, hands and fingers, clipped to the person"""
    H, W = m.shape
    c = np.zeros((H, W), np.uint8)

    def line(a, b, v=200):
        cv2.line(c, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), v, 1, cv2.LINE_AA)

    def rings(a, b, half, step):
        a, b = np.array(a, float), np.array(b, float)
        d = b - a
        L = np.hypot(*d)
        if L < 4:
            return
        pp = np.array([-d[1], d[0]]) / L
        for k in range(int(L / step) + 1):
            q = a + d * (k * step / L)
            line(q - pp * half, q + pp * half, 170)

    P = {k: (v[0] * W, v[1] * H) for k, v in tr["pose"].items()}
    sw = np.hypot(P[11][0] - P[12][0], P[11][1] - P[12][1])
    for _, y, z in ((12, 14, 16), (11, 13, 15)):
        rings(P[y], P[z], sw * 0.1, sw * 0.07)
    for h in tr["hands"]:
        Q = [(a * W, b * H) for a, b in h]
        pl = np.hypot(Q[0][0] - Q[9][0], Q[0][1] - Q[9][1])
        for f in (1, 5, 9, 13, 17):
            for j in range(f, f + 3 if f != 1 else 3):
                rings(Q[j], Q[j + 1], pl * 0.14, pl * 0.1)
    if tr["face"]:
        F = [(a * W, b * H) for a, b in tr["face"]]
        for k, e in enumerate(FACE_MESH):
            if k % 2 == 0:
                line(F[e.start], F[e.end], 150)
    return c.astype(np.float32) / 255 * cv2.erode((m > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8))


def outline(lift, m):
    """white line drawing: thin edges, contour lines that follow the forms, and the silhouette"""
    l = cv2.cvtColor(lift, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    f = cv2.bilateralFilter(l, 9, 0.1, 7)
    e = cv2.Canny((cv2.GaussianBlur(f, (0, 0), 1.3) * 255).astype(np.uint8), 22, 58, L2gradient=True)
    e = cv2.GaussianBlur(cv2.dilate(e, np.ones((2, 2), np.uint8)).astype(np.float32) / 255, (0, 0), 0.6).clip(0, 1) * 1.15
    s = cv2.GaussianBlur(f, (0, 0), 7)
    bands = s * 11
    dist = np.abs(bands - np.round(bands))
    gx, gy = cv2.Sobel(bands, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(bands, cv2.CV_32F, 0, 1, ksize=3) / 8
    iso = (1 - dist / (np.hypot(gx, gy) * 0.8 + 1e-4)).clip(0, 1) * 0.65
    inner = cv2.erode((m > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    edge = cv2.morphologyEx((m > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((4, 4), np.uint8)).astype(np.float32)
    return np.maximum(np.maximum(e, iso) * inner, edge).clip(0, 1)


def look_px(look, lift, m, tr, SPR, BOX, colour):
    if look == "neg":
        l = cv2.cvtColor(lift, cv2.COLOR_BGR2RGB).astype(float).mean(2) / 255
        neg = 1 - l
        return np.dstack([neg ** 0.7, neg ** 0.8 * 0.98, neg ** 1.6 * 0.55]).clip(0, 1) ** 0.85 * 255
    if look.startswith("wire"):
        g = np.maximum(outline(lift, m), mesh(m, tr))
        g = np.dstack([g, g, g])
    else:
        g = xray(lift, m, tr, SPR, BOX)
        g = np.dstack([g * 1.06, g * 1.01, g * 0.97]).clip(0, 1)   # the cool tone
    back = colour if look.endswith("colour") else np.zeros(3)
    return 255 - (255 - back) * (1 - g)                            # screened over the colour, or over black


# ------------------------------------------------------ the camera ------------------------------------------------------
def out_curve(u):
    """monotone cubic through the measured samples, as the approved camera does it"""
    if u <= 0:
        return 0.0
    if u >= 1:
        return 1.0
    p = OUT_CURVE
    x = u * (len(p) - 1)
    i = int(x)
    f = x - i
    m = lambda k: (p[min(k + 1, len(p) - 1)] - p[max(k - 1, 0)]) / (min(k + 1, len(p) - 1) - max(k - 1, 0))
    return (2 * f**3 - 3 * f**2 + 1) * p[i] + (f**3 - 2 * f**2 + f) * m(i) + (-2 * f**3 + 3 * f**2) * p[i + 1] + (f**3 - f**2) * m(i + 1)


def zoom_at(t, punch):
    """t = seconds after --start. Scale moves in log space, so equal steps read as equal zoom."""
    t0, land = FIRST_FLASH / RUN_FPS, LAND / RUN_FPS
    if t < t0:
        return 1.0
    if t < land:
        return float(np.exp(np.log(punch) * ((t - t0) / (land - t0)) ** ACCEL))
    held = 0.5 - 0.5 * np.cos(np.pi * min((t - land) / 1.3, 1))
    z = punch * CREEP ** held
    if t < land + HOLD:
        return float(z)
    top = punch * CREEP ** (0.5 - 0.5 * np.cos(np.pi * min(HOLD / 1.3, 1)))
    return float(np.exp(np.log(top) * (1 - out_curve((t - land - HOLD) / RETURN))))


def apply_zoom(frame, z, anchor):
    if z <= 1.0001:
        return frame
    ax, ay = anchor
    M = np.float32([[z, 0, ax * (1 - z)], [0, z, ay * (1 - z)]])
    return cv2.warpAffine(frame, M, frame.shape[1::-1], flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------ the render ------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Put the x-ray flicker on a talking head clip.")
    ap.add_argument("clip")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the clip where the effect starts (default 0)")
    ap.add_argument("--colour", default="#1C3A7C", help="the colour behind every second x-ray and line drawing, as a hex code")
    ap.add_argument("--punch", type=float, default=PUNCH, help="how far the camera goes in (default 1.5; 1 = no camera move)")
    ap.add_argument("--out", default=None, help="where to save the video (default renders/<clip>-xray.mp4)")
    a = ap.parse_args()

    clip = Path(a.clip).expanduser().resolve()
    if not clip.exists():
        die(f"no file at {clip}")
    hexa = a.colour.lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", hexa):
        die(f"--colour must be a hex code like #1C3A7C, not {a.colour}")
    colour = np.array([int(hexa[4:6], 16), int(hexa[2:4], 16), int(hexa[0:2], 16)], float)   # BGR
    out = Path(a.out).expanduser().resolve() if a.out else HERE / "renders" / f"{clip.stem}-xray.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    fps, rate, seconds, hdr = probe(clip)
    vf = decode_filter(rate, hdr)
    W, H = clip_size(clip, vf)
    run = (FIRST_FLASH + FLASH_STEP * len(LOOKS)) / RUN_FPS
    if seconds is not None and a.start + run > seconds:
        die(f"the effect needs {run:.1f} s from --start {a.start:g}, and the clip is {seconds:.1f} s long")
    print(f"Clip: {W}x{H}, {fps:.3f} fps" + (f", {seconds:.1f} s" if seconds else ""), flush=True)

    # which look lands on which frame of this clip's frame rate
    spans = [(a.start + (FIRST_FLASH + FLASH_STEP * k) / RUN_FPS, a.start + (FIRST_FLASH + FLASH_STEP * k + 1) / RUN_FPS, look) for k, look in enumerate(LOOKS)]
    look_of = {}
    for t0, t1, look in spans:
        for n in range(int(np.ceil(t0 * fps - 1e-6)), max(int(np.ceil(t0 * fps - 1e-6)) + 1, int(np.ceil(t1 * fps - 1e-6)))):
            look_of[n] = look
    margin = max(3, int(round(fps / 8)))
    first, last = min(look_of) - margin, max(look_of) + margin

    print("Tracking the body...", flush=True)
    T = tracker()
    SPR, BOX = load_parts()
    scale = min(1.0, WORK_SIZE / max(W, H))
    ws = (int(round(W * scale)), int(round(H * scale)))
    work, TR = {}, {}
    for n, fr in enumerate(frames(clip, vf, (W, H))):
        if n > last:
            break
        if n >= first:
            small = cv2.resize(fr, ws, interpolation=cv2.INTER_AREA) if scale < 1 else fr.copy()
            work[n] = small
            TR[n] = track(T, small, lift_of(small))
    if not any(t["pose"] for t in TR.values()):
        die("no person found at that point in the clip. Use a clip with one person, chest up, facing the camera.")

    print("Building the flashes...", flush=True)
    flashes, sheet = {}, []
    aspect = ws[0] / ws[1]
    for n, look in sorted(look_of.items()):
        if n not in work:
            continue
        small = work[n]
        lift = lift_of(small)
        m = person_mask(T, lift)
        tr = {"pose": nearest(TR, n, lambda t: t.get("pose")), "face": nearest(TR, n, lambda t: t.get("face")), "hands": hands_at(TR, n, aspect)}
        px = look_px(look, lift, m, tr, SPR, BOX, colour).clip(0, 255)
        al = m[..., None]
        done = (small * (1 - al) + px * al).clip(0, 255).astype(np.uint8)
        flashes[n] = cv2.resize(done, (W, H), interpolation=cv2.INTER_CUBIC) if scale < 1 else done
        if not sheet or sheet[-1][0] != look or n - sheet[-1][1] > 1:
            sheet.append((look, n, done))

    # the camera: about the nose bridge, never so far that the head leaves the top of the frame
    mid = sorted(look_of)[len(look_of) // 2]
    face = nearest(TR, mid, lambda t: t.get("face"))
    P = nearest(TR, mid, lambda t: t.get("pose"))
    if face:
        anchor = ((face[168][0]) * W, (face[168][1]) * H)
    else:
        anchor = ((P[11][0] + P[12][0]) / 2 * W, (P[11][1] + P[12][1]) / 2 * H * 0.6)
    punch = a.punch
    if punch > 1:
        rows = np.where((person_mask(T, lift_of(work[mid])) > 0.5).any(1))[0]
        head_top = rows[0] / ws[1] * H if len(rows) else 0
        room = (anchor[1] - HEADROOM * H) / max(anchor[1] - head_top, 1)
        punch = max(min(punch, room / CREEP), min(1.12, punch))

    print("Rendering...", flush=True)
    enc = subprocess.Popen([FF, "-v", "error", "-y", "-nostdin", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", rate, "-i", "-", "-i", str(clip),
                            "-map", "0:v:0", "-map", "1:a?", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                            "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                            "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    count = 0
    for n, fr in enumerate(frames(clip, vf, (W, H))):
        pic = flashes.get(n, fr)
        z = zoom_at(n / fps - a.start, punch) if punch > 1 else 1.0
        enc.stdin.write(apply_zoom(pic, z, anchor).tobytes())
        count = n + 1
    enc.stdin.close()
    if enc.wait() != 0:
        die("the video encoder failed")

    # a check picture of every look, for Claude Code to open and look at
    if sheet:
        tw = 360
        th = int(round(tw * ws[1] / ws[0]))
        cols = 5
        rows_n = (len(sheet) + cols - 1) // cols
        canvas = np.zeros((rows_n * th, cols * tw, 3), np.uint8)
        for i, (_, _, im) in enumerate(sheet):
            canvas[(i // cols) * th:(i // cols + 1) * th, (i % cols) * tw:(i % cols + 1) * tw] = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out.with_name(out.stem + "-check.jpg")), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Done: {out} ({count} frames, {len(flashes)} flash frames, camera to {punch:.2f}x)")


if __name__ == "__main__":
    main()
