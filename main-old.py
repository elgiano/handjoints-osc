import cv2
import mediapipe as mp
import numpy as np
import argparse
from pythonosc.udp_client import SimpleUDPClient

hands = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils


def send_osc_hands(client, hands):
    for (n_hand, hand) in enumerate(hands):
        coords = [(joint.x, joint.y) for joint in hand.landmark]
        coords = [c for joint in coords for c in joint]
        # coords = [item for sublist in list for item in sublist]
        # coords = [c for joint in hand.landmark for c in (joint.x, joint.y)]
        client.send_message("/handjoints/", [n_hand] + coords)


def send_osc_landmarks(client, mp_res):
    hands = mp_res.multi_hand_landmarks
    # print([c.classification) for c in mp_res.multi_handedness])
    msg = [c for hand in hands for joint in hand.landmark
        for c in (joint.x, joint.y)]
    client.send_message("/handjoints", msg)


def main(host, port, confidence):
    show_numbers = False

    osc_client = SimpleUDPClient(host, port)

    mp_hands = hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=confidence)
    cap = cv2.VideoCapture(0)

    # Create an empty black image
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    black_bg = np.zeros((height, width, 3), dtype=np.uint8)

    title = "HandJointsOSC"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE |
                    cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO)

    print("Starting. Press 'n' to toggle joint numbers.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        black_bg.fill(0)

        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image_rgb)

        if results.multi_hand_landmarks:
            send_osc_landmarks(osc_client, results)
            # Draw landmarks
            for (n_hand, hand) in enumerate(results.multi_hand_landmarks):
                drawing_utils.draw_landmarks(black_bg, hand, hands.HAND_CONNECTIONS)
                # Draw joint number on the black_bg image
                if show_numbers:
                    for (n_joint, landmark) in enumerate(hand.landmark):
                        x, y = landmark.x, landmark.y
                        joint_coords = (int(x * width), int(y * height))
                        cv2.putText(black_bg, str(n_joint), joint_coords,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(title, black_bg)
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

