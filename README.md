# VELOCITY_TRACKR v9

Fast object tracker with Kalman prediction, trajectory trail, and speed HUD.

## Run from source
pip install -r requirements.txt
python tracker.py

## Build EXE (no Python needed to run)
Double-click build.bat
→ EXE appears in dist/VELOCITY_TRACKR.exe

## Controls
| Key | Action |
|-----|--------|
| S   | Select target |
| R   | Reset tracker |
| T   | Cycle tracker type (CSRT → KCF → MOSSE) |
| Q   | Quit |

## What's new in v9
- Kalman filter predicts position when object moves fast or briefly disappears
- Trajectory trail shows movement history
- Speed meter in px/s with colour warning at high speed
- Prediction arrow shows where object is heading
- T key to switch tracker on the fly