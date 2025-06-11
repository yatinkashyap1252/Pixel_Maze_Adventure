# Pixel Fox Maze Adventure

Pixel Fox Maze Adventure is a fun and interactive color tracking maze game built with Python, Pygame, and OpenCV. In this game, you control a pixel-art fox character by moving a blue object in front of your webcam. The goal is to navigate the fox through a procedurally generated maze and reach the green flag to win.

## Features

- Real-time color tracking using OpenCV to detect a blue object.
- Procedurally generated maze using Depth-First Search algorithm.
- Pixel art style graphics for the maze, fox character, and UI.
- Movement cooldown to prevent rapid moves.
- Game stats including time elapsed and moves count.
- UI panels displaying game controls, camera view, and instructions.
- Win and game over screens with pixel art decorations.
- Keyboard controls to restart or quit the game.

## Requirements

- Python 3.x
- Pygame
- OpenCV (cv2)
- NumPy

## Installation

1. Install Python 3.x from [python.org](https://www.python.org/downloads/).

2. Install required Python packages using pip:

```bash
pip install pygame opencv-python numpy
```

## How to Play

1. Run the game script:

```bash
python color_maze_game.py
```

2. Hold a blue object (e.g., a blue pen or paper) in front of your webcam.

3. Move the blue object to guide the fox through the maze.

4. The fox moves one cell at a time and cannot jump over grass walls.

5. Reach the green flag (end point) to win the game.

6. Press **R** to restart the game.

7. Press **Q** to quit the game.

## Controls

- **R**: Restart the game with a new maze.
- **Q**: Quit the game.

## How It Works

- The game captures video frames from your webcam using OpenCV.
- It detects the largest blue object in the frame by filtering the HSV color space.
- The centroid of the detected blue object determines the direction for the fox to move.
- The maze is generated randomly each game using a depth-first search algorithm.
- The fox is represented as pixel art and moves smoothly within the maze grid.
- The UI displays the camera view, instructions, game stats, and movement cooldown.

## Notes

- You may need to adjust the HSV color range in the script if your blue object is not detected properly.
- Ensure your webcam is connected and accessible.
- The game window size is 1250x650 pixels for optimal UI layout.

## License

This project is open source and free to use.

---

Enjoy guiding your pixel fox through the colorful maze adventure!
