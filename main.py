import cv2
import numpy as np
import argparse
from pythonosc.udp_client import SimpleUDPClient

import mediapipe as mp
from mediapipe.tasks.python import vision as mpv
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
RunningMode = mpv.RunningMode
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


def draw_landmarks(image, detection_result, show_numbers):
    hand_landmarks = detection_result.hand_landmarks
    for landmarks in hand_landmarks:
        # Draw the hand landmarks.
        ext_landmarks = landmark_pb2.NormalizedLandmarkList()
        ext_landmarks.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(image, ext_landmarks,
                                                  HAND_CONNECTIONS)
        # Draw joint number on the black_bg image
        if show_numbers:
            height, width = image.shape[:2]
            for (n_joint, landmark) in enumerate(landmarks):
                x, y = landmark.x, landmark.y
                joint_coords = (int(x * width), int(y * height))
                cv2.putText(image, str(n_joint), joint_coords,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)


def send_osc_hands(client, mp_res):
    for (i, hand) in enumerate(mp_res.hand_landmarks):
        msg = [i]
        isRight = mp_res.handedness[i][0].index > 0
        coords = [c for j in hand for c in (j.x, 1 - j.y)]
        msg.append(isRight)
        msg += coords
        client.send_message("/handjoints", msg)


def send_osc_landmarks(client, mp_res):
    msg = []
    for (i, hand) in enumerate(mp_res.hand_landmarks):
        msg.append(i)
        isRight = mp_res.handedness[i][0].index > 0
        coords = [c for j in hand for c in (j.x, 1 - j.y)]
        msg.append(isRight)
        msg += coords

    client.send_message("/handjoints", msg)


def main(host, port, confidence):

    show_numbers = False
    osc_client = SimpleUDPClient(host, port)

    # Create an empty black image
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bg = np.zeros((height, width, 3), dtype=np.uint8)

    title = "HandJointsOSC"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE |
                    cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Create mediapipe detector
    # mp_hands = hands.Hands(static_image_mode=False, max_num_hands=2,
    #                        min_detection_confidence=confidence)
    # base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = mpv.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_tracking_confidence=confidence,
    )
    mp_hands = mpv.HandLandmarker.create_from_options(options)

    print("Starting. Press 'n' to toggle joint numbers.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        bg.fill(0)
        image = cv2.flip(image, 1)
        results = mp_hands.detect(mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        if results:
            send_osc_landmarks(osc_client, results)
            # Draw landmarks
            draw_landmarks(bg, results, show_numbers)

        cv2.imshow(title, bg)

        # Wait for a key press event or 1 ms to allow window to refresh
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            show_numbers = not show_numbers
        # Check if the window is closed
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int,
                        help="send OSC to this port")
    parser.add_argument("--host", default="127.0.0.1", type=str,
                        help="send OSC to this host")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                        help="minimum detection confidence threshold")
    args = parser.parse_args()

    main(args.host, args.port, args.confidence)
