import cv2
import numpy as np


def segment_final_fix(video_path, frame_number, output_path="final_vertical.jpg", show_result=True, debug_save=True):
    # -----------------------------
    # Read exact frame
    # -----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video:", video_path)
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: Target frame not found.")
        return

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_img, w_img = frame.shape[:2]

    # Crops (bench/scoreboard areas)
    top_crop = int(h_img * 0.30)
    bottom_crop = int(h_img * 0.25)  # IMPORTANT: bench strip is bigger than 0.18

    def apply_crops(mask):
        m = mask.copy()
        m[:top_crop, :] = 0
        m[h_img - bottom_crop:, :] = 0
        return m

    # Kernels
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_vclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 9))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 31))

    # -----------------------------
    # Build a GOOD pitch mask (two grass ranges)
    # -----------------------------
    grass1 = cv2.inRange(hsv, (35, 35, 35), (95, 255, 210))
    grass2 = cv2.inRange(hsv, (35, 20, 20), (95, 255, 140))
    pitch  = cv2.bitwise_or(grass1, grass2)

    pitch = apply_crops(pitch)
    pitch = cv2.morphologyEx(pitch, cv2.MORPH_OPEN, k5, iterations=1)

    # pick largest component NOW (before strong close)
    cnts, _ = cv2.findContours(pitch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pitch_main = np.zeros_like(pitch)
    if cnts:
        c_big = max(cnts, key=cv2.contourArea)
        cv2.drawContours(pitch_main, [c_big], -1, 255, thickness=cv2.FILLED)

    # optional: gentle close AFTER
    pitch_main = cv2.morphologyEx(pitch_main, cv2.MORPH_CLOSE, k9, iterations=1)
    pitch_area = cv2.dilate(pitch_main, k9, iterations=1)


    # -----------------------------
    # Line mask (remove line spam for white/ball)
    # -----------------------------
    # --- stronger line extraction: 0/45/90/135 degrees ---
    def line_kernel(length, angle_deg):
        k = np.zeros((length, length), np.uint8)
        c = length // 2
        if angle_deg == 0:      # horizontal
            k[c, :] = 1
        elif angle_deg == 90:   # vertical
            k[:, c] = 1
        elif angle_deg == 45:   # diag /
            for i in range(length):
                k[i, length - 1 - i] = 1
        elif angle_deg == 135:  # diag \
            for i in range(length):
                k[i, i] = 1
        return k

    line_white = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))
    line_white = apply_crops(line_white)
    line_white = cv2.bitwise_and(line_white, pitch_area)

    # multi-angle extraction
    lines_only = np.zeros_like(line_white)
    for ang in (0, 45, 90, 135):
        k = line_kernel(41, ang)
        extracted = cv2.morphologyEx(line_white, cv2.MORPH_OPEN, k, iterations=1)
        lines_only = cv2.bitwise_or(lines_only, extracted)

    # ALSO include long H/V components (but don't overwrite)
    line_h = cv2.morphologyEx(line_white, cv2.MORPH_OPEN, kh, iterations=1)
    line_v = cv2.morphologyEx(line_white, cv2.MORPH_OPEN, kv, iterations=1)
    lines_only = cv2.bitwise_or(lines_only, line_h)
    lines_only = cv2.bitwise_or(lines_only, line_v)

    # widen for removal
    lines_only_big   = cv2.dilate(lines_only, k5, iterations=2)
    lines_only_small = cv2.dilate(lines_only, k3, iterations=1)

    def gate_on_pitch(mask, lines_mask):
        m = apply_crops(mask)
        m = cv2.bitwise_and(m, pitch_area)
        m = cv2.bitwise_and(m, cv2.bitwise_not(lines_mask))
        return m


    # -----------------------------
    # Color masks (GK / Green / White / Ref)
    # -----------------------------

    # GK (orange) – tighter hue AND require higher saturation/value (reduces ads / boards)
    mask_gk = cv2.inRange(hsv, (5, 80, 80), (30, 255, 255))
    mask_gk = gate_on_pitch(mask_gk, lines_only_small)
    mask_gk = cv2.morphologyEx(mask_gk, cv2.MORPH_OPEN, k3, iterations=1)
    mask_gk = cv2.morphologyEx(mask_gk, cv2.MORPH_CLOSE, k_vclose, iterations=2)


    # --- GREEN KIT (stable version) ---
    mask_green = cv2.inRange(hsv, (40, 50, 170), (60, 170, 255))

    mask_green = gate_on_pitch(mask_green, lines_only_big)

    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, k3, iterations=1)


    cnts, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask_green)

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        ar = h / max(w,1)

        # Remove grass blobs
        if area < 50 or area > 20000:
            continue
        if ar < 0.5 or ar > 50.5:
            continue

        cv2.drawContours(clean, [c], -1, 255, cv2.FILLED)

    mask_green = clean


    # White kit – tighten saturation upper bound (stops light-green / glare turning “white”)
    mask_white = cv2.inRange(hsv, (0, 0, 190), (180, 55, 255))
    mask_white = gate_on_pitch(mask_white, lines_only_big)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, k3, iterations=1)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, k_vclose, iterations=2)

    # use a smaller close so you don't stitch line fragments into tall components
    k_close_player = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, k_close_player, iterations=1)

    # remove lines again after close
    mask_white = cv2.bitwise_and(mask_white, cv2.bitwise_not(lines_only))

    # Ref (dark kit)
    mask_ref = cv2.inRange(hsv, (0, 0, 0), (180, 220, 95))
    mask_ref   = gate_on_pitch(mask_ref,   lines_only_big)
    mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_OPEN, k3, iterations=1)
    mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_CLOSE, k_vclose, iterations=2)


    # -----------------------------
    # Build PERSON candidates from (green|white|ref|gk)
    # -----------------------------
    candidates = cv2.bitwise_or(mask_green, mask_white)
    candidates = cv2.bitwise_or(candidates, mask_ref)
    candidates = cv2.bitwise_or(candidates, mask_gk)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, k_vclose, iterations=2)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, k3, iterations=1)

    # Remove pitch lines AGAIN (morphology can reconnect them into blobs)
    candidates = cv2.bitwise_and(candidates, cv2.bitwise_not(lines_only))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, k3, iterations=1)


    # -----------------------------
    # Ball candidates (fixed)
    # -----------------------------
    H, S, V = cv2.split(hsv)

    Vg = cv2.bitwise_and(V, pitch_main)   # <-- FIX: use pitch_main (exists)

    k_ball = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    V_tophat = cv2.morphologyEx(Vg, cv2.MORPH_TOPHAT, k_ball)

    bright  = cv2.inRange(V_tophat, 25, 255)
    low_sat = cv2.inRange(S, 0, 150)

    ball_cand = cv2.bitwise_and(bright, low_sat)

    ball_cand = gate_on_pitch(ball_cand, lines_only_small)
    ball_cand = cv2.morphologyEx(ball_cand, cv2.MORPH_OPEN, k3, iterations=1)



    # -----------------------------
    # Debug saves
    # -----------------------------
    if debug_save:
        cv2.imwrite("dbg_pitch_main.png", pitch_main)
        cv2.imwrite("dbg_pitch_area.png", pitch_area)
        cv2.imwrite("dbg_lines_only.png", lines_only)
        cv2.imwrite("dbg_gk.png", mask_gk)
        cv2.imwrite("dbg_green.png", mask_green)
        cv2.imwrite("dbg_white.png", mask_white)
        cv2.imwrite("dbg_ref.png", mask_ref)
        cv2.imwrite("dbg_candidates.png", candidates)
        cv2.imwrite("dbg_ball.png", ball_cand)
        print("Wrote debug masks: dbg_*.png")

    out = frame.copy()

    # -----------------------------
    # Draw BALL (best circular blob)
    # -----------------------------
    best_ball = None
    best_score = -1.0
    cnts_ball, _ = cv2.findContours(ball_cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts_ball:
        area = cv2.contourArea(c)
        if area < 6 or area > 400:
            continue

        x, y, w, h = cv2.boundingRect(c)
        ar = w / max(h, 1)
        if ar < 0.6 or ar > 1.6:
            continue

        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circ = 4.0 * np.pi * area / (peri * peri)

        # keep only reasonably circular blobs
        if circ < 0.45:
            continue

        score = circ * (1.0 + area / 400.0)
        if score > best_score:
            best_score = score
            best_ball = (x, y, w, h)

    if best_ball is not None:
        x, y, w, h = best_ball
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(out, "Ball", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # -----------------------------
    # 7) Draw PLAYERS/REF/GK from candidate contours
    #    classify by per-bbox mask ratios
    # -----------------------------
    cnts, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_id = 1
    white_id = 1

    MIN_AREA = 180
    MAX_AREA = 45000
    MIN_H = 18
    MIN_AR = 0.85
    MAX_AR = 6     #  long thin shapes
    MIN_W = 12    #  rejects very thin vertical posts

    best_gk = None
    best_gk_score = -1.0

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cnt_area = cv2.contourArea(c)

        extent = cnt_area / float(w * h + 1e-6)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = cnt_area / float(hull_area + 1e-6)

        # blob constraints (tune if needed)
        if extent < 0.18:      # lines are tiny extent
            continue
        if solidity < 0.45:    # arcs/lines are low solidity
            continue
        if min(w, h) < 6:      # very thin shapes
            continue

        # reject thin line-like shapes
        if w <= 6 or (h / max(w, 1)) > 7.0:
            continue

        # prevent giant boxes (merged blobs)
        if w > int(0.10 * w_img) or h > int(0.35 * h_img):
            continue

        area = w * h

        if area < MIN_AREA or area > MAX_AREA:
            continue
        if h < MIN_H:
            continue
        if (y + h) > (h_img - bottom_crop):
            continue
        if w > int(0.30 * w_img) or h > int(0.70 * h_img):
            continue

        ar = h / max(w, 1)
        if ar < MIN_AR or ar > MAX_AR:
            continue
        if w < MIN_W:
            continue

        roi_area = w * h
        if roi_area <= 0:
            continue

        # Reject anything that is mostly line pixels
        line_r = cv2.countNonZero(lines_only[y:y+h, x:x+w]) / roi_area
        if line_r > 0.04:
            continue


        gk_r = cv2.countNonZero(mask_gk[y:y+h, x:x+w]) / roi_area
        gr_r = cv2.countNonZero(mask_green[y:y+h, x:x+w]) / roi_area
        wh_r = cv2.countNonZero(mask_white[y:y+h, x:x+w]) / roi_area
        rf_r = cv2.countNonZero(mask_ref[y:y+h, x:x+w]) / roi_area

        # --- GK: collect candidates, but don't draw yet ---
        # Require stronger gk ratio than before to avoid orange boards
        if gk_r >= 0.06 and 1.2 <= ar <= 5.5 and area >= 400:
            gk_score = gk_r * (1.0 + area / 5000.0)
            if gk_score > best_gk_score:
                best_gk_score = gk_score
                best_gk = (x, y, w, h)
            # Don't classify this contour as team/white/ref yet; skip for now
            continue

        # --- Team / Ref classification ---
        label = None
        color = None


        # Prefer green/white over ref if either is present
        if gr_r >= 0.05 or wh_r >= 0.05:
            if gr_r > wh_r:
                # green box size cap (prevents huge green boxes)
                if w > int(0.06 * w_img) or h > int(0.28 * h_img):
                    continue
                label, color = f"Green {green_id}", (0, 255, 0)
                green_id += 1
            else:
                label, color = f"White {white_id}", (255, 255, 255)
                white_id += 1

        elif rf_r >= 0.08:
            label, color = "Ref", (0, 0, 0)

        if label is not None:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, label, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw ONLY the best GK (max 1)
    if best_gk is not None:
        x, y, w, h = best_gk
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 140, 255), 2)
        cv2.putText(out, "GK", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)


    cv2.imwrite(output_path, out)

    if show_result:
        cv2.imshow("Final Fix", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Processed:", output_path)

if __name__ == "__main__":
    segment_final_fix(
        video_path="08fd33_4.mp4",
        frame_number=700,
        output_path="final_vertical.jpg",
        show_result=True,
        debug_save=True
    )
