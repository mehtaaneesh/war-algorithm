import numpy as np
import cv2
import os

cap = cv2.VideoCapture("anti_uav_video.mp4")

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame.astype(np.float32) / 255.0
    frames.append(frame)


def rgb_to_hsv_manual(img):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    V = np.max(img, axis=2)
    min_rgb = np.min(img, axis=2)
    S = np.where(V == 0, 0, (V - min_rgb) / V)

    H = np.zeros_like(V)

    mask = V == R
    H[mask] = (60 * ((G - B)/(V - min_rgb + 1e-6)))[mask]

    mask = V == G
    H[mask] = (60 * (2 + (B - R)/(V - min_rgb + 1e-6)))[mask]

    mask = V == B
    H[mask] = (60 * (4 + (R - G)/(V - min_rgb + 1e-6)))[mask]

    H = np.mod(H, 360)
    return H, S, V


def estimate_pdf(values, bins):
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    return hist, bin_edges

H_vals, S_vals, V_vals = [], [], []

for i in range(30):  # initial frames
    H, S, V = rgb_to_hsv_manual(frames[i])
    H_vals.extend(H.flatten())
    S_vals.extend(S.flatten())
    V_vals.extend(V.flatten())

pH, h_bins = estimate_pdf(H_vals, 50)
pS, s_bins = estimate_pdf(S_vals, 50)
pV, v_bins = estimate_pdf(V_vals, 50)

def lookup_prob(value, hist, bins):
    idx = np.digitize(value, bins) - 1
    idx = np.clip(idx, 0, len(hist)-1)
    return hist[idx]

def detect_anomaly(H, S, V, pH, h_bins, pS, s_bins, pV, v_bins, eps=1e-3):
    PH = lookup_prob(H, pH, h_bins)
    PS = lookup_prob(S, pS, s_bins)
    PV = lookup_prob(V, pV, v_bins)

    P = PH * PS * PV
    return (P < eps).astype(np.uint8)

def connected_components(binary_img):
    visited = np.zeros_like(binary_img)
    components = []

    H, W = binary_img.shape

    for i in range(H):
        for j in range(W):
            if binary_img[i,j] == 1 and not visited[i,j]:
                stack = [(i,j)]
                comp = []

                while stack:
                    x,y = stack.pop()
                    if x<0 or y<0 or x>=H or y>=W:
                        continue
                    if visited[x,y] or binary_img[x,y]==0:
                        continue
                    visited[x,y] = 1
                    comp.append((x,y))
                    for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        stack.append((x+dx,y+dy))

                components.append(comp)
    return components

def bounding_box(component):
    xs = [p[0] for p in component]
    ys = [p[1] for p in component]
    return min(xs), min(ys), max(xs), max(ys)

def center_of_mass(component):
    xs = np.array([p[0] for p in component])
    ys = np.array([p[1] for p in component])
    return xs.mean(), ys.mean()

def gaussian_mask(shape, center, sigma):
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')

    return np.exp(-((X-center[0])**2 + (Y-center[1])**2) / (2*sigma**2))


def optical_flow(prev, curr, center, win=5):
    cx, cy = int(center[0]), int(center[1])
    Ix = curr[cx-win:cx+win, cy-win:cy+win] - prev[cx-win:cx+win, cy-win:cy+win]
    Iy = curr[cx-win:cx+win, cy-win:cy+win].T - prev[cx-win:cx+win, cy-win:cy+win].T
    It = curr[cx-win:cx+win, cy-win:cy+win] - prev[cx-win:cx+win, cy-win:cy+win]

    A = np.stack((Ix.flatten(), Iy.flatten()), axis=1)
    b = -It.flatten()

    flow, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return flow

prev_gray = None
prev_center = None
trajectory = []

h, w, _ = (frames[0] * 255).astype(np.uint8).shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    'output_tracking.mp4',
    fourcc,
    20.0,
    (w, h)
)


for t in range(len(frames)):

    frame = frames[t]
    gray = np.mean(frame, axis=2)

    H, S, V = rgb_to_hsv_manual(frame)
    anomaly_map = detect_anomaly(
        H, S, V,
        pH, h_bins,
        pS, s_bins,
        pV, v_bins
    )

    components = connected_components(anomaly_map)

    print(
        f"Frame {t:04d} | "
        f"Anomaly pixels: {np.sum(anomaly_map)} | "
        f"Components: {len(components)}"
    )

    if len(components) == 0:
        vis = (frame * 255).astype(np.uint8)

        cv2.putText(
            vis,
            f"Frame {t} - No Detection",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        out.write(vis)

        trajectory.append(None)
        prev_gray = gray
        prev_center = None
        continue

    largest_component = max(components, key=len)

    center = center_of_mass(largest_component)

    print(f"  → Tracked center: ({center[0]:.1f}, {center[1]:.1f})")

    vis = (frame * 255).astype(np.uint8)

    x_min, y_min, x_max, y_max = bounding_box(largest_component)

    cv2.rectangle(
        vis,
        (y_min, x_min),
        (y_max, x_max),
        (0, 255, 0),
        2
    )

    cv2.circle(
        vis,
        (int(center[1]), int(center[0])),
        5,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        vis,
        f"Frame {t}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    out.write(vis)

    if prev_gray is not None and prev_center is not None:
        flow = optical_flow(prev_gray, gray, prev_center)
        center = (
            center[0] + flow[0],
            center[1] + flow[1]
        )

    trajectory.append(center)

    prev_gray = gray
    prev_center = center

out.release()
print("Saved output video: output_tracking.mp4")


