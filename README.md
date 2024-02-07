#handjoints-osc

Real-time hand tracking with MediaPipe. Sends data over OSC. Written in Python.

### Installation

Recommended: make a virtual environment

  python -m venv .venv
  source .venv/bin/activate

Install requirements:

  pip install -r requirements

Arguments:

  $ python main.py --help

  usage: main.py [-h] [--host HOST] [--confidence CONFIDENCE] port

  positional arguments:
    port                  send OSC to this port

  options:
    -h, --help            show this help message and exit
    --host HOST           send OSC to this host
    --confidence CONFIDENCE, -c CONFIDENCE
                          minimum detection confidence threshold

### OSC format

- path: `/handjoints i *i* ...f`
- [0] number of detected hands
- [1:numHands] handedness for each detected hand
- [numHands+1:..] x and y coordinates for each joint for each hand

The program detects maximum 2 hands, each hand has 21 joints, and each joint 2 coordinates.

Arguments are all in a single list, starting with the number of hands, then handedness for each hand, and following with x and y coordinates for all joints of one hand, and then the joint of each other hand.

  [nHands, ...handedness, ...coordsHand0, ...cordsHand1]
  coords: [j0x, j0y, j1x, j1y, j2x, j2y, ...]

If only one hand is detected, numHands + handedness + coords (21 * 2) gives 44 values.
If two hands are detected, there are two handedness values, so 1 + 2 + 42 + 42 = 87 values.
