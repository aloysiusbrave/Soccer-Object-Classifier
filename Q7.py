import cv2
import numpy as np

def segment_final_fix(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    
    # 1. SETUP BACKGROUND SUBTRACTION
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    
    # Fast forward
    start_frame = max(0, frame_number - 60)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    frame_target = None
    fg_mask = None
    
    print("Processing...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        mask = back_sub.apply(frame, learningRate=0.005) # Increased rate slightly
        
        if idx == frame_number:
            frame_target = frame
            fg_mask = mask
            break
        idx += 1
            
    if frame_target is None: return

    # 2. CREATE DETECTION MASKS
    hsv = cv2.cvtColor(frame_target, cv2.COLOR_BGR2HSV)
    
    # A. Motion Mask (Cleaned)
    _, motion_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    
    # B. GK Force Mask (Red/Orange) - Catches him even if static
    # Hue 0-20 (Red-Orange) and 160-180 (Red wrap-around)
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 70, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_gk_1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_gk_2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_gk_static = cv2.bitwise_or(mask_gk_1, mask_gk_2)
    
    # C. Ball Mask (White & Small)
    # High Value (>180), Low Saturation (<30)
    mask_ball_static = cv2.inRange(hsv, (0, 0, 180), (180, 30, 255))
    
    # COMBINE ALL DETECTION SOURCES
    # Motion OR GK OR Ball
    combined_mask = cv2.bitwise_or(motion_mask, mask_gk_static)
    # (Optional: Add ball mask to detection if ball is stationary, but usually ball moves)
    
    # ROI: Kill Top 30% (Crowd)
    h_img, w_img = combined_mask.shape
    combined_mask[0:int(h_img*0.3), :] = 0 
    
    # Morphology to connect body parts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 15))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # 3. FIND CONTOURS
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame_target.copy()
    
    player_id = 1
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        roi = hsv[y:y+h, x:x+w]
        if roi.size == 0: continue

        # --- LOGIC 1: THE BALL ---
        # CHECK THIS BEFORE SIZE FILTERING
        # Ball is small (10-300px), Square-ish ratio, and WHITE
        ratio = float(h)/w
        if 10 < area < 400 and 0.5 < ratio < 1.5:
            # Check for White pixels in this small box
            white_pixels = cv2.countNonZero(cv2.inRange(roi, (0, 0, 160), (180, 50, 255)))
            if white_pixels > (area * 0.3): # If 30% of box is white
                cv2.rectangle(output, (x, y), (x+w, y+h), (0,0,255), 2)
                cv2.putText(output, "Ball", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                continue # Stop processing this box

        # --- PRE-FILTERING FOR PLAYERS ---
        if h < 25: continue # Now we filter small noise
        
        # --- PLAYER CLASSIFICATION ---
        
        # 1. DEFINE PIXEL COUNTS (Strict brightness control)
        
        # GK (Red/Orange)
        votes_gk = cv2.countNonZero(cv2.inRange(roi, (0, 70, 70), (20, 255, 255))) + \
                   cv2.countNonZero(cv2.inRange(roi, (160, 70, 70), (180, 255, 255)))
                   
        # REF (Black)
        # Value < 50 (Very Dark). Saturation < 50 (Not dark green/red).
        votes_ref = cv2.countNonZero(cv2.inRange(roi, (0, 0, 0), (180, 50, 60)))
        
        # NEON GREEN (The "Anti-Grass" Mask)
        # Value > 140. This forces the pixel to be BRIGHT. 
        # Grass is usually Value 80-100.
        votes_neon = cv2.countNonZero(cv2.inRange(roi, (35, 50, 140), (90, 255, 255)))
        
        # WHITE
        # Value > 160, Saturation < 40.
        votes_white = cv2.countNonZero(cv2.inRange(roi, (0, 0, 160), (180, 40, 255)))
        
        label = None
        color = None
        
        # 2. DECISION TREE
        
        # A. GK?
        if votes_gk > 20:
             label, color = "GK", (0, 140, 255) # Orange box
             
        # B. Ref? (Check black pixels)
        elif votes_ref > 30:
             label, color = "Ref", (0, 0, 0) # Black box
             
        # C. White vs Green Showdown
        # We check who has MORE pixels to decide.
        elif votes_neon > 10 or votes_white > 10:
            if votes_neon > votes_white:
                label, color = "Green", (0, 255, 0)
            else:
                label, color = "White", (255, 255, 255)
        
        # D. Ghost Box Killer
        # If it didn't match ANY of the above (no neon, no white, no black, no orange),
        # it is just a box of grass. We do nothing.
        
        # 3. DRAW
        if label:
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
            if label in ["White", "Green"]:
                cv2.putText(output, f"{label} {player_id}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                player_id += 1
            else:
                cv2.putText(output, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Final Fix", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("final_vertical.jpg", output)
    print("Processed.")

if __name__ == "__main__":
    segment_final_fix("08fd33_4.mp4", 700)