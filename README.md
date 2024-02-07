# handjoints-osc

Real-time hand tracking that sends joint coordinates over OSC. Written in Python, using Google [MediaPipe](https://developers.google.com/mediapipe).

## Installation

Installing via [pipx](https://github.com/pypa/pipx) is recommended, because it installs in an isolated environment.

  pipx install handjoints-osc

It's of course also possible to install via pip:

  pip install handjoints-osc

## Usage

  $ handjoints-osc --help

  usage: handjoints-osc [-h] [--host HOST] [--confidence CONFIDENCE] port

  positional arguments:
    port                  send OSC to this port

  options:
    -h, --help            show this help message and exit
    --host HOST           send OSC to this host (default: localhost)
    --confidence CONFIDENCE, -c CONFIDENCE
                          minimum detection confidence threshold (default: 0.5)

For example, to start the program and send joints coordinates to SuperCollider, which typically listens for OSC on port 57120:

  handjoints-osc 57120

When hands are detected, SuperCollider will start receiving OSC messages with path "/handjoints-osc".
To know which value corresponds to which joint, pressing "n" will display joint numbers on the tracking window.

## OSC format

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

## Development

Recommended: make a virtual environment

  python -m venv .venv
  source .venv/bin/activate

Install requirements:

  pip install -r requirements

