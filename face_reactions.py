import cv2
import math
import os
import numpy as np
import mediapipe as mp

SHOW_DEBUG = True      # live numbers for tuning (set False when done)
IMAGE_FOLDER = ""      # optional: r"C:\path\to\your\images"
PANEL = 480
W, H = 640, 480

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


IMAGES = {
    "ellie":  (["ellie"], []),
    "apple":  (["apple"], ["smil"]),      # tongue-out apple (slightly open mouth)
    "smile":  (["smil"], []),             # smiling apple
    "wojak":  (["hamp", "hamster", "wojak"], []),
    "man":    (["speed"], []),
    "sad":    (["sad"], []),
    "kanye":  (["kanye"], []),
    "kratos": (["kratos"], []),
}

def search_folders():
    home = os.path.expanduser("~")
    folders = []
    if IMAGE_FOLDER:
        folders.append(IMAGE_FOLDER)
    folders.append(os.path.dirname(os.path.abspath(__file__)))
    folders.append(os.getcwd())
    for base in (home, os.path.join(home, "OneDrive")):
        folders.append(os.path.join(base, "Documents", "face_reactions"))
        folders.append(os.path.join(base, "Documents"))
        folders.append(os.path.join(base, "Downloads"))
        folders.append(os.path.join(base, "Desktop"))
        folders.append(os.path.join(base, "Pictures"))
    seen, result = set(), []
    for f in folders:
        if f not in seen and os.path.isdir(f):
            seen.add(f)
            result.append(f)
    return result

FOLDERS = search_folders()

def find_image(keywords, excluded):
    for folder in FOLDERS:
        for fname in sorted(os.listdir(folder)):
            low = fname.lower()
            if (low.endswith(EXTS)
                    and any(k in low for k in keywords)
                    and not any(x in low for x in excluded)):
                return os.path.join(folder, fname)
    return None

def read_image(path):
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return img

def fit_to_panel(img):
    h, w = img.shape[:2]
    scale = min(PANEL / w, PANEL / h)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.zeros((PANEL, PANEL, 3), np.uint8)
    y = (PANEL - img.shape[0]) // 2
    x = (PANEL - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas

def placeholder(name):
    img = np.full((PANEL, PANEL, 3), 80, np.uint8)
    cv2.putText(img, "missing: " + name, (30, PANEL // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

images = {}
for name, (keys, excl) in IMAGES.items():
    path = find_image(keys, excl)
    img = read_image(path) if path else None
    if img is not None:
        images[name] = fit_to_panel(img)
        print("Loaded", name, "->", path)
    else:
        images[name] = placeholder(name)
        print("NOT FOUND:", name, "(file name should contain one of", keys, ")")

print("\nFolders searched:")
for f in FOLDERS:
    print("  ", f)

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands_det = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def pdist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def to_px(landmarks):
    return [(p.x * W, p.y * H) for p in landmarks]

cap = cv2.VideoCapture(0)
current, pending, count = "ellie", "ellie", 0

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read from camera.")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (W, H))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_res = face_mesh.process(rgb)
    hand_res = hands_det.process(rgb)

    hands = []
    if hand_res.multi_hand_landmarks:
        for hl in hand_res.multi_hand_landmarks:
            hands.append(to_px(hl.landmark))
            if SHOW_DEBUG:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

    label = "ellie"
    heart = False
    ear = False
    thumb_gap = index_gap = ear_d = -1.0
    eye_open = mouth_open = mouth_w = smile = asym = 0.0
    face_found = bool(face_res.multi_face_landmarks)

    if len(hands) == 2:
        h1, h2 = hands
        size = (pdist(h1[0], h1[9]) + pdist(h2[0], h2[9])) / 2
        if size > 0:
            thumb_gap = pdist(h1[4], h2[4]) / size
            index_gap = pdist(h1[8], h2[8]) / size
            index_above = (h1[8][1] + h2[8][1]) / 2 < (h1[4][1] + h2[4][1]) / 2
            if thumb_gap < 0.8 and index_gap < 0.8 and index_above:
                heart = True

    if face_found:
        lm = face_res.multi_face_landmarks[0].landmark

        face_w = dist(lm[234], lm[454])
        eye_open = (dist(lm[159], lm[145]) + dist(lm[386], lm[374])) / 2 / face_w
        mouth_open = dist(lm[13], lm[14]) / face_w
        mouth_w = dist(lm[61], lm[291]) / face_w
        mid_y = (lm[13].y + lm[14].y) / 2
        corners_y = (lm[61].y + lm[291].y) / 2
        smile = (mid_y - corners_y) / face_w       # + smiling, - frowning
        asym = abs(lm[61].y - lm[291].y) / face_w

        left_ear = (lm[234].x * W, lm[234].y * H)
        right_ear = (lm[454].x * W, lm[454].y * H)
        face_px = pdist(left_ear, right_ear)
        for hd in hands:
            palm = (sum(hd[i][0] for i in (0, 5, 9, 13, 17)) / 5,
                    sum(hd[i][1] for i in (0, 5, 9, 13, 17)) / 5)
            d = min(pdist(palm, left_ear), pdist(palm, right_ear)) / face_px
            if ear_d < 0 or d < ear_d:
                ear_d = d
            if d < 0.5:
                ear = True

    if heart:
        label = "kratos"
    elif ear:
        label = "kanye"
    elif face_found:
        if mouth_open > 0.12:
            label = "wojak"
        elif eye_open < 0.02 and mouth_w < 0.36:
            label = "man"
        elif smile < -0.015:
            label = "sad"
        elif asym > 0.025:
            label = "ellie"
        elif smile > 0.02:
            label = "smile"
        elif mouth_open > 0.04:
            label = "apple"
        else:
            label = "ellie"

    if SHOW_DEBUG:
        t1 = "eye %.3f open %.3f width %.3f smile %.3f asym %.3f" % (
            eye_open, mouth_open, mouth_w, smile, asym)
        t2 = "hands %d  thumbGap %.2f  indexGap %.2f  earDist %.2f" % (
            len(hands), thumb_gap, index_gap, ear_d)
        cv2.putText(frame, t1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, t2, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, label, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    if label == pending:
        count += 1
    else:
        pending, count = label, 1
    if count >= 4:
        current = pending

    combined = np.hstack((frame, images[current]))
    cv2.imshow("Face Reactions (press q to quit)", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
