import cv2
import math
import time
import os
import urllib.request
import numpy as np
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter

BEND_THRESHOLD = 1.25
STOP_DELAY = 1.0
SMOOTHING_FRAMES = 5

pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()

FREQS = {
    "do": 261.63, "re": 293.66, "mi": 329.63, "fa": 349.23,
    "so": 392.00, "la": 440.00, "ti": 493.88, 
    "do_high": 523.25, "re_high": 587.33, "mi_high": 659.25, 
    "fa_high": 698.46, "so_high": 783.99, "la_high": 880.00, "ti_high": 987.77
}

def generate_wave(freq):
    t = np.linspace(0, 5.0, int(44100 * 5.0), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
    audio = (wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(np.column_stack((audio, audio))))

SOUNDS = {k: generate_wave(v) for k, v in FREQS.items()}
audio_channel = pygame.mixer.Channel(0)

GESTURE_MAP = {
    (True, False, False, False): "do",       # 1000 
    (True, True, False, False): "re",        # 1100 
    (True, True, True, False): "mi",         # 1110 
    (True, True, True, True): "fa",          # 1111 
    (False, True, True, True): "so",         # 0111 
    (False, False, True, True): "la",        # 0011 
    (False, False, False, True): "do_high",  # 0001 
    (True, False, False, True): "re_high",   # 1001 
    (True, True, False, True): "mi_high",    # 1101 
    (False, True, False, True): "so_high",   # 0101 

    (False, True, False, False): "ti",       # 0100 
    (False, True, True, False): "fa_high",   # 0110 
    (True, False, True, False): "la_high",   # 1010 
    (True, False, True, True): "ti_high"     # 1011 
}

'''    The version above is a special edition designed to perform the music mice on venus in optimal condition, I do suggest to use the standard version below for other music
GESTURE_MAP = {
    (True, False, False, False): "do",       
    (True, True, False, False): "re",        
    (True, True, True, False): "mi",         
    (True, True, True, True): "fa",          
    (False, True, False, False): "so",       
    (False, True, True, False): "la",        
    (False, True, True, True): "ti",         
    (False, False, True, True): "do_high",   
    (False, False, False, True): "re_high",  
    (True, False, False, True): "mi_high",   
    (True, True, False, True): "fa_high",    
    (True, False, True, True): "so_high",    
    (False, True, False, True): "la_high",   
    (False, False, True, False): "ti_high"
}
'''

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path)

detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO, num_hands=2,
    min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5
))

def calc_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def choose_camera():
    available_cameras = []
    
    for i in range(4):
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            if ret:
                available_cameras.append(i)
        test_cap.release()

    if not available_cameras:
        print("Whoops, couldn't find any cameras!")
        exit()

    print(f"\nFound {len(available_cameras)} available camera(s)")
    for cam_idx in available_cameras:
        if cam_idx == 0:
            print(f"  [{cam_idx}] Default camera")
        else:
            print(f"  [{cam_idx}] External camera")

    if len(available_cameras) == 1:
        selected_idx = available_cameras[0]
    else:
        choice = input(f"\nWhich camera do you want to use? (Hit Enter for default [{available_cameras[0]}]): ")
        if choice.isdigit() and int(choice) in available_cameras:
            selected_idx = int(choice)
        else:
            selected_idx = available_cameras[0]
            print(f"Invalid input. Sticking with the default camera [{selected_idx}].")

    return cv2.VideoCapture(selected_idx)

cap = choose_camera()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

window_name = "Virtual Violin"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

is_fullscreen = True
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_rx, prev_ry = 0, 0
last_motion_time = 0
current_note = None

gesture_buffer = []

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    curr_t = time.time()
    
    ai_process_img = cv2.resize(img, (600, int(600 * (h / w))))
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(ai_process_img, cv2.COLOR_BGR2RGB))
    results = detector.detect_for_video(mp_img, int(curr_t * 1000))

    active_note_name = None
    is_rest_signal = False 

    if results.hand_landmarks:
        hands = sorted([(lms[0].x, lms) for lms in results.hand_landmarks], key=lambda x: x[0])
        left = hands[0][1] if len(hands) >= 1 else None
        right = hands[1][1] if len(hands) >= 2 else None

        if left:
            for lm in left: cv2.circle(img, (int(lm.x*w), int(lm.y*h)), 10, (255, 120, 0), -1)
            
            dist_thumb_to_pinky = calc_dist(left[4], left[17])
            palm_width = calc_dist(left[5], left[17]) 
            
            if dist_thumb_to_pinky > palm_width * 1.4: 
                is_rest_signal = True
                cv2.putText(img, "REST (Thumb Out)", (int(left[4].x*w), int(left[4].y*h)-20), 1, 4.0, (0, 0, 255), 6)
                gesture_buffer.clear() 

            f_s = calc_dist(left[8], left[0]) > calc_dist(left[6], left[0])
            f_z = calc_dist(left[12], left[0]) > calc_dist(left[10], left[0])
            f_w = calc_dist(left[16], left[0]) > calc_dist(left[14], left[0])
            f_x = calc_dist(left[20], left[0]) > calc_dist(left[18], left[0])

            raw_str = f"Raw: [{'1' if f_s else '0'}{'1' if f_z else '0'}{'1' if f_w else '0'}{'1' if f_x else '0'}]"
            cv2.putText(img, raw_str, (int(left[0].x*w)-50, int(left[0].y*h)+80), 1, 3.0, (0, 255, 255), 4)

            if not is_rest_signal:
                current_gesture = (f_s, f_z, f_w, f_x)
                
                gesture_buffer.append(current_gesture)
                if len(gesture_buffer) > SMOOTHING_FRAMES:
                    gesture_buffer.pop(0)
                
                stable_gesture = Counter(gesture_buffer).most_common(1)[0][0]
                
                stable_str = f"Stable: [{'1' if stable_gesture[0] else '0'}{'1' if stable_gesture[1] else '0'}{'1' if stable_gesture[2] else '0'}{'1' if stable_gesture[3] else '0'}]"
                cv2.putText(img, stable_str, (int(left[0].x*w)-50, int(left[0].y*h)+150), 1, 3.0, (0, 255, 0), 4)

                active_note_name = GESTURE_MAP.get(stable_gesture, None)

        if right:
            for lm in right: cv2.circle(img, (int(lm.x*w), int(lm.y*h)), 10, (0, 255, 0), -1)
            move_dist = math.hypot(right[0].x - prev_rx, right[0].y - prev_ry)
            if move_dist > 0.005: 
                last_motion_time = curr_t
            prev_rx, prev_ry = right[0].x, right[0].y

    bowing = (curr_t - last_motion_time < STOP_DELAY)
    
    if bowing and not is_rest_signal and active_note_name:
        if current_note != active_note_name:
            audio_channel.play(SOUNDS[active_note_name], loops=-1, fade_ms=50)
            current_note = active_note_name
      
        cv2.rectangle(img, (0,0), (w,h), (0, 255, 0), 20) 
    else:
        if current_note:
            fade_time = 80 if is_rest_signal else 200
            audio_channel.fadeout(fade_time)
            current_note = None

    txt = f"PLAYING: {active_note_name}" if current_note else "REST / STOPPED"
    txt_color = (0, 255, 0) if current_note else (0, 0, 255)
    cv2.putText(img, txt, (50, 120), 1, 5.0, txt_color, 6)
    
    cv2.imshow(window_name, img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == 27:
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720) 

cap.release()
pygame.quit()
cv2.destroyAllWindows()