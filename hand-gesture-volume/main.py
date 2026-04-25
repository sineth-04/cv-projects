import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ── MediaPipe setup ──────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ── Windows Volume Control setup ─────────────────────────────
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices._dev.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume.GetVolumeRange()[:2]

# ── Webcam setup ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)                              # mirror the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # convert to RGB for MediaPipe
        results = hands.process(rgb_frame)                      # run hand detection

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get screen dimensions
                h, w, _ = frame.shape

                # Get thumb tip (4) and index finger tip (8) coordinates
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                # Convert from 0-1 range to actual pixel positions
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                # Draw circles on thumb and index tip
                cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)   # blue dot
                cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), -1)   # blue dot

                # Draw a line between them
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 3)

                # Calculate distance between thumb and index tip
                distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

                # Map distance (20 to 200 pixels) to volume range (min to max)
                # Think of it like mapping a ruler to a dial
                vol = vol_min + (distance - 20) * (vol_max - vol_min) / (200 - 20)
                vol = max(vol_min, min(vol_max, vol))   # clamp so it doesn't go out of range

                # Set the system volume
                volume.SetMasterVolumeLevel(vol, None)

                # Show distance and volume % on screen
                vol_percent = int((distance - 20) / (200 - 20) * 100)
                vol_percent = max(0, min(100, vol_percent))
                cv2.putText(frame, f'Volume: {vol_percent}%', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Hand Volume Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()