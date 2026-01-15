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
    m = np.min(img, axis=2)
    S = np.where(V == 0, 0, (V - m) / (V + 1e-6))

    H = np.zeros_like(V)

    mask = V == R
    H[mask] = 60 * ((G - B) / (V - m + 1e-6))[mask]

    mask = V == G
    H[mask] = 60 * (2 + (B - R) / (V - m + 1e-6))[mask]

    mask = V == B
    H[mask] = 60 * (4 + (R - G) / (V - m + 1e-6))[mask]

    H = np.mod(H, 360)
    return H, S, V

def estimate_pdf(values, bins=50):
    hist, edges = np.histogram(values, bins=bins, density=True)
    return hist, edges

H_vals, S_vals, V_vals = [], [], []

for i in range(30):
    H, S, V = rgb_to_hsv_manual(frames[i])
    H_vals.extend(H.flatten())
    S_vals.extend(S.flatten())
    V_vals.extend(V.flatten())

pH, h_bins = estimate_pdf(H_vals)
pS, s_bins = estimate_pdf(S_vals)
pV, v_bins = estimate_pdf(V_vals)

def lookup_prob(v, hist, bins):
    idx = np.digitize(v, bins) - 1
    idx = np.clip(idx, 0, len(hist)-1)
    return hist[idx]

def detect_anomaly(H, S, V, eps=1e-3):
    P = lookup_prob(H, pH, h_bins) \
      * lookup_prob(S, pS, s_bins) \
      * lookup_prob(V, pV, v_bins)
    return (P < eps).astype(np.uint8)

def connected_components(bin_img):
    H, W = bin_img.shape
    visited = np.zeros_like(bin_img)
    components = []

    for i in range(H):
        for j in range(W):
            if bin_img[i,j] and not visited[i,j]:
                stack = [(i,j)]
                comp = []
                while stack:
                    x,y = stack.pop()
                    if x<0 or y<0 or x>=H or y>=W:
                        continue
                    if visited[x,y] or not bin_img[x,y]:
                        continue
                    visited[x,y] = 1
                    comp.append((x,y))
                    for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        stack.append((x+dx,y+dy))
                components.append(comp)
    return components

def bounding_box(comp):
    xs = [p[0] for p in comp]
    ys = [p[1] for p in comp]
    return min(xs), min(ys), max(xs), max(ys)

def center_of_mass(comp):
    xs = np.array([p[0] for p in comp])
    ys = np.array([p[1] for p in comp])
    return xs.mean(), ys.mean()

def optical_flow(prev, curr, center, win=5):
    x,y = int(center[0]), int(center[1])
    if x-win<0 or y-win<0 or x+win>=prev.shape[0] or y+win>=prev.shape[1]:
        return np.array([0.0,0.0])

    Ix = curr[x-win:x+win, y-win:y+win] - prev[x-win:x+win, y-win:y+win]
    Iy = curr[x-win:x+win, y-win:y+win].T - prev[x-win:x+win, y-win:y+win].T
    It = curr[x-win:x+win, y-win:y+win] - prev[x-win:x+win, y-win:y+win]

    A = np.stack((Ix.flatten(), Iy.flatten()), axis=1)
    b = -It.flatten()

    flow, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return flow

def compute_motion_metrics(traj):
    speeds, directions, vertical = [], [], []
    for i in range(1, len(traj)):
        if traj[i] is None or traj[i-1] is None:
            continue
        x1,y1 = traj[i-1]
        x2,y2 = traj[i]
        vx,vy = x2-x1, y2-y1
        speeds.append(np.sqrt(vx**2 + vy**2))
        directions.append(np.arctan2(vy,vx))
        vertical.append(vx)
    return np.array(speeds), np.array(directions), np.array(vertical)

def direction_change_rate(dirs):
    if len(dirs)<2:
        return 0
    d = np.abs(np.diff(dirs))
    d = np.minimum(d, 2*np.pi-d)
    return np.mean(d)

def classify_target(traj):
    s,d,v = compute_motion_metrics(traj)
    if len(s)<5:
        return "Unknown"
    if np.var(s)>2.0 or direction_change_rate(d)>0.6:
        return "Bird / Noise"
    if np.mean(s)>4.0 and np.mean(v)>0.5:
        return "Attack Drone"
    if np.mean(s)<=4.0 and direction_change_rate(d)<0.3:
        return "Recon Drone"
    return "Unknown"

def print_threat_stats(traj, frame_idx):
    s, d, v = compute_motion_metrics(traj)
    if len(s) < 5:
        return

    avg_speed = np.mean(s)
    speed_var = np.var(s)
    dir_change = direction_change_rate(d)
    label = classify_target(traj)

    print(
        f"[Frame {frame_idx:04d}] "
        f"Threat: {label} | "
        f"Avg Speed: {avg_speed:.2f} | "
        f"Speed Var: {speed_var:.2f} | "
        f"Dir Change: {dir_change:.2f}"
    )


h,w,_ = frames[0].shape
out = cv2.VideoWriter("output_tracking_threat.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      20,(w,h))

trajectory = []
prev_gray = None
prev_center = None

for t,frame in enumerate(frames):
    gray = np.mean(frame, axis=2)

    H,S,V = rgb_to_hsv_manual(frame)
    anomaly = detect_anomaly(H,S,V)
    comps = connected_components(anomaly)

    vis = (frame*255).astype(np.uint8)

    if comps:
        comp = max(comps, key=len)
        center = center_of_mass(comp)

        if prev_gray is not None and prev_center is not None:
            flow = optical_flow(prev_gray, gray, prev_center)
            center = (center[0]+flow[0], center[1]+flow[1])

        x1,y1,x2,y2 = bounding_box(comp)
        cv2.rectangle(vis,(y1,x1),(y2,x2),(0,255,0),2)
        cv2.circle(vis,(int(center[1]),int(center[0])),5,(0,0,255),-1)

        trajectory.append(center)
        prev_center = center
    else:
        trajectory.append(None)
        prev_center = None

    if t > 30:
        recent_traj = trajectory[-30:]
        label = classify_target(recent_traj)

        cv2.putText(
            vis,
            f"Target: {label}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        if t % 10 == 0:
            print_threat_stats(recent_traj, t)

    cv2.putText(vis,f"Frame {t}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    out.write(vis)
    prev_gray = gray

out.release()
print("Saved video: output_tracking_threat.mp4")

xs = [p[1] for p in trajectory if p is not None]
ys = [p[0] for p in trajectory if p is not None]

plt.figure(figsize=(6,6))
plt.plot(xs, ys, '-r')
plt.gca().invert_yaxis()
plt.title("Tracked Trajectory")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.savefig("trajectory_plot_threat.jpg")
plt.show()

print("Saved trajectory plot: trajectory_plot_threat.jpg")
