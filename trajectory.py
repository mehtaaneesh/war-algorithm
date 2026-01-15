import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("anti_uav_video.mp4")

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame.astype(np.float32) / 255.0)

cap.release()

def rgb_to_hsv_manual(img):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    V = np.max(img, axis=2)
    min_rgb = np.min(img, axis=2)
    S = np.where(V == 0, 0, (V - min_rgb) / (V + 1e-6))

    H = np.zeros_like(V)

    mask = V == R
    H[mask] = 60 * ((G - B)/(V - min_rgb + 1e-6))[mask]

    mask = V == G
    H[mask] = 60 * (2 + (B - R)/(V - min_rgb + 1e-6))[mask]

    mask = V == B
    H[mask] = 60 * (4 + (R - G)/(V - min_rgb + 1e-6))[mask]

    return np.mod(H, 360), S, V

def estimate_pdf(values, bins):
    hist, edges = np.histogram(values, bins=bins, density=True)
    return hist, edges

H_vals, S_vals, V_vals = [], [], []

for i in range(30):
    H, S, V = rgb_to_hsv_manual(frames[i])
    H_vals.extend(H.flatten())
    S_vals.extend(S.flatten())
    V_vals.extend(V.flatten())

pH, h_bins = estimate_pdf(H_vals, 50)
pS, s_bins = estimate_pdf(S_vals, 50)
pV, v_bins = estimate_pdf(V_vals, 50)

def lookup_prob(val, hist, bins):
    idx = np.digitize(val, bins) - 1
    idx = np.clip(idx, 0, len(hist)-1)
    return hist[idx]

def detect_anomaly(H, S, V, eps=1e-3):
    P = (
        lookup_prob(H, pH, h_bins) *
        lookup_prob(S, pS, s_bins) *
        lookup_prob(V, pV, v_bins)
    )
    return (P < eps).astype(np.uint8)

def connected_components(binary):
    visited = np.zeros_like(binary)
    comps = []
    h, w = binary.shape

    for i in range(h):
        for j in range(w):
            if binary[i,j] and not visited[i,j]:
                stack = [(i,j)]
                comp = []
                while stack:
                    x,y = stack.pop()
                    if x<0 or y<0 or x>=h or y>=w:
                        continue
                    if visited[x,y] or binary[x,y]==0:
                        continue
                    visited[x,y] = 1
                    comp.append((x,y))
                    for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        stack.append((x+dx,y+dy))
                comps.append(comp)
    return comps

def center_of_mass(comp):
    xs = np.array([p[0] for p in comp])
    ys = np.array([p[1] for p in comp])
    return xs.mean(), ys.mean()

def bounding_box(comp):
    xs = [p[0] for p in comp]
    ys = [p[1] for p in comp]
    return min(xs), min(ys), max(xs), max(ys)

h, w, _ = frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    "output_tracking_with_trajectory.mp4",
    fourcc,
    20.0,
    (w, h)
)

trajectory = []
prev_center = None

for t, frame in enumerate(frames):

    gray = np.mean(frame, axis=2)

    H, S, V = rgb_to_hsv_manual(frame)
    anomaly = detect_anomaly(H, S, V)

    components = connected_components(anomaly)

    vis = (frame * 255).astype(np.uint8)

    if len(components) > 0:
        comp = max(components, key=len)
        center = center_of_mass(comp)
        trajectory.append(center)

        x1,y1,x2,y2 = bounding_box(comp)
        cv2.rectangle(vis,(y1,x1),(y2,x2),(0,255,0),2)

        for i in range(1,len(trajectory)):
            if trajectory[i-1] and trajectory[i]:
                p1 = (int(trajectory[i-1][1]), int(trajectory[i-1][0]))
                p2 = (int(trajectory[i][1]), int(trajectory[i][0]))
                cv2.line(vis,p1,p2,(255,0,0),2)

        cv2.circle(
            vis,
            (int(center[1]), int(center[0])),
            5,(0,0,255),-1
        )
    else:
        trajectory.append(None)

    cv2.putText(
        vis,f"Frame {t}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,(255,255,255),2
    )

    out.write(vis)

out.release()
print("Saved video: output_tracking_with_trajectory.mp4")

traj = np.array([p for p in trajectory if p is not None])

plt.figure(figsize=(6,6))
plt.plot(traj[:,1], traj[:,0], 'r.-')
plt.gca().invert_yaxis()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Drone Trajectory")
plt.grid()
plt.savefig("trajectory_plot.jpg", dpi=200)
plt.close()

print("Saved trajectory plot: trajectory_plot.jpg")
