environment:
	python3 -m venv .venv

setup:
	pip install mediapipe
	pip install wheel
	pip install pyautogui

run:
	python3 subway_surfer.py