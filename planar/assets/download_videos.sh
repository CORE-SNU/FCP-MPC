#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/videos"

gdown https://drive.google.com/uc?id=1KxY24LMWEua92TYmKjKY4eZdkohg3kGm -O "$SCRIPT_DIR/videos/eth.avi"
gdown https://drive.google.com/uc?id=1F5WP2blQMyHAq0t--NKKeZCb8GdjEMvk -O "$SCRIPT_DIR/videos/hotel.avi"
gdown https://drive.google.com/uc?id=1VmbQ6-3diBZvYXg8LrjU4mYCbYcZ0Fub -O "$SCRIPT_DIR/videos/zara1.avi"
gdown https://drive.google.com/uc?id=1520YOv7_ropMtPDjX6__ppq3cxT10Ton -O "$SCRIPT_DIR/videos/zara2.avi"
gdown https://drive.google.com/uc?id=1VbyPPzC9aUPVQaODT4gyQfzg6CYiFkU2 -O "$SCRIPT_DIR/videos/univ.avi"


