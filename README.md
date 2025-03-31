# Smart Face Tracking Pan/Tilt Project

A Pi 5 project using Raspi Module 3 camera that detects faces in real time and automatically moves a pan/tilt to follow them. It streams the live feed over RTSP so you can view it from any device on your local network.

## Features
- Real-time face detection with OpenCV Haar Cascades running on the Pi 5's ARM Cortex-A76 CPU
- Auto pan/tilt tracking with Adafruit PCA9685 servos
- RTSP video stream via GStreamer
- Manual control: use 'wasd' keys for pan/tilt, 'c' to center, 't' to toggle tracking, 'q' to quit

## Hardware
- Raspberry Pi 5 (anything older may not have enough CPU power)
- Raspberry Pi Camera (v3)
- PCA9685 servo controller + 2 SG92R servos

## Setup
1. Install dependencies:
> sudo apt update<br>
sudo apt install python3-pip python3-smbus ffmpeg libsm6 libxext6<br>
pip3 install adafruit-circuitpython-pca9685 adafruit-circuitpython-servokit sshkeyboard opencv-contrib-python

2. Enable I2C:
> sudo raspi-config  # Enable I2C under Interface Options in the Pi settings, then reboot

3. Run the script:
> python3 main.py
