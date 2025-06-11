import cv2
import numpy as np
import pygame
import sys
import random
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1250  # Increased window width for better UI layout
WINDOW_HEIGHT = 650  # Increased window height
CELL_SIZE = 40
GRID_WIDTH = 20  # Fixed grid width
GRID_HEIGHT = 15  # Fixed grid height
MAZE_WIDTH = GRID_WIDTH * CELL_SIZE
MAZE_HEIGHT = GRID_HEIGHT * CELL_SIZE
MAZE_LEFT = 20  # Left margin for maze
MAZE_TOP = 20  # Top margin for maze
UI_PANEL_WIDTH = WINDOW_WIDTH - MAZE_WIDTH - MAZE_LEFT - 20  # Width of UI panel
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
DARK_GREEN = (0, 100, 0)  # Dark green for grass
LIGHT_GREEN = (50, 205, 50)  # Light green for grass highlights
YELLOW = (255, 255, 0)  # Yellow color for direction indicator
LIGHT_BLUE = (173, 216, 230)  # Light blue for UI elements
GOLD = (255, 215, 0)  # Gold color for start point
GRAY = (128, 128, 128)  # Gray for background elements
ORANGE = (255, 165, 0)  # Orange for fox character
BROWN = (139, 69, 19)  # Brown for fox details

# Load and scale background image (create a simple gradient background)
def create_background():
    background = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    for y in range(WINDOW_HEIGHT):
        # Create a gradient from dark gray to light gray
        color_value = 20 + (y * 40 // WINDOW_HEIGHT)
        color = (color_value, color_value, color_value + 10)
        pygame.draw.line(background, color, (0, y), (WINDOW_WIDTH, y))
    return background

# Create rounded rectangle function
def draw_rounded_rect(surface, rect, color, radius=10):
    """Draw a rectangle with rounded corners"""
    rect = pygame.Rect(rect)
    
    # Draw the main rectangle without corners
    pygame.draw.rect(surface, color, rect.inflate(-radius*2, 0))
    pygame.draw.rect(surface, color, rect.inflate(0, -radius*2))
    
    # Draw the four corners
    pygame.draw.circle(surface, color, (rect.left + radius, rect.top + radius), radius)
    pygame.draw.circle(surface, color, (rect.right - radius, rect.top + radius), radius)
    pygame.draw.circle(surface, color, (rect.left + radius, rect.bottom - radius), radius)
    pygame.draw.circle(surface, color, (rect.right - radius, rect.bottom - radius), radius)

# Create a function to draw a panel with a title
def draw_panel(surface, rect, color, title=None, title_color=WHITE, border_radius=10, alpha=200):
    # Create a surface with per-pixel alpha
    panel_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    
    # Draw the panel background with transparency
    draw_rounded_rect(panel_surface, pygame.Rect(0, 0, rect.width, rect.height), 
                     (*color, alpha), radius=border_radius)
    
    # Draw the title if provided
    if title:
        font = pygame.font.Font(None, 28)
        title_text = font.render(title, True, title_color)
        title_rect = title_text.get_rect(midtop=(rect.width // 2, 10))
        panel_surface.blit(title_text, title_rect)
        
        # Draw a separator line below the title
        pygame.draw.line(panel_surface, title_color, 
                        (20, title_rect.bottom + 5), 
                        (rect.width - 20, title_rect.bottom + 5), 2)
# Draw pixel art grass wall
def draw_pixel_grass(surface, cell_size):
    """Draw a pixel art grass wall on the given surface"""
    # Base dark green background
    pygame.draw.rect(surface, DARK_GREEN, (0, 0, cell_size, cell_size))
    
    # Create pixel pattern for grass
    pixel_size = cell_size // 8  # 8x8 pixel grid
    
    # Define the pixel pattern (1 = light green pixel, 0 = keep dark green)
    grass_pattern = [
        [0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0]
    ]
    
    # Draw the pixels according to the pattern
    for y in range(8):
        for x in range(8):
            if grass_pattern[y][x] == 1:
                pygame.draw.rect(surface, LIGHT_GREEN, 
                               (x * pixel_size, y * pixel_size, 
                                pixel_size, pixel_size))

# Draw pixel art fox character
def draw_pixel_fox(surface, size):
    """Draw a pixel art fox character on the given surface"""
    # Create a transparent surface for the fox
    fox_surface = pygame.Surface((size, size), pygame.SRCALPHA)
    
    # Define the pixel pattern for the fox (8x8 grid)
    # 0 = transparent, 1 = orange (body), 2 = white (details), 3 = brown (details), 4 = black (eyes)
    fox_pattern = [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 3, 1, 1, 1, 1, 3, 1],
        [1, 4, 1, 1, 1, 1, 4, 1],
        [1, 1, 2, 1, 1, 2, 1, 1],
        [1, 1, 1, 2, 2, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0]
    ]
    
    # Color mapping
    colors = {
        0: (0, 0, 0, 0),  # Transparent
        1: ORANGE,        # Orange for body
        2: WHITE,         # White for details
        3: BROWN,         # Brown for ears
        4: BLACK          # Black for eyes
    }
    
    # Draw the pixels according to the pattern
    pixel_size = size // 8
    for y in range(8):
        for x in range(8):
            color = colors[fox_pattern[y][x]]
            if fox_pattern[y][x] != 0:  # Skip transparent pixels
                pygame.draw.rect(fox_surface, color, 
                               (x * pixel_size, y * pixel_size, 
                                pixel_size, pixel_size))
    
    return fox_surface    
    # Blit the panel onto the main surface
    surface.blit(panel_surface, rect)

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Color Tracking Maze Game')
clock = pygame.time.Clock()

# Create background
background = create_background()

# Load fonts
title_font = pygame.font.Font(None, 48)
instruction_font = pygame.font.Font(None, 24)
status_font = pygame.font.Font(None, 32)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device.")
    sys.exit()

# Define the color range to track (blue)
# You may need to adjust these values based on your object's color
lower_color = np.array([100, 100, 100])  # Blue in HSV
upper_color = np.array([140, 255, 255])  # Blue in HSV

# Player variables
player_x = 1
player_y = 1
move_cooldown = 0  # Cooldown timer to prevent too rapid movement
MOVE_DELAY = 10    # Number of frames to wait between moves

# Initialize game stats
moves_count = 0
start_time = pygame.time.get_ticks()
game_time = 0

# Generate maze using Depth-First Search algorithm
def generate_maze(width, height):
    # Create a grid filled with walls
    maze = [['#' for _ in range(width)] for _ in range(height)]
    
    # Define the recursive function
    def carve_passages(x, y):
        maze[y][x] = ' '  # Mark current cell as passage
        
        # Define possible directions: right, down, left, up
        directions = [(2, 0), (0, 2), (-2, 0), (0, -2)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == '#':
                # Carve passage between current cell and next cell
                maze[y + dy//2][x + dx//2] = ' '
                carve_passages(nx, ny)
    
    # Start from a random point (must be odd coordinates)
    start_x = 1
    start_y = 1
    carve_passages(start_x, start_y)
    
    # Set start and end points
    maze[1][1] = 'S'
    maze[height-2][width-2] = 'E'
    
    return maze

# Generate the maze
maze = generate_maze(GRID_WIDTH, GRID_HEIGHT)

# Game state
game_over = False
win = False

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_r:  # Reset game
                maze = generate_maze(GRID_WIDTH, GRID_HEIGHT)
                player_x, player_y = 1, 1
                move_cooldown = 0
                game_over = False
                win = False
                moves_count = 0
                start_time = pygame.time.get_ticks()
                game_time = 0
            elif event.key == K_q:  # Quit game
                pygame.quit()
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()
    
    # Process video frame for color tracking
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture video frame.")
        break
    
    # Flip the frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the specified color
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Target position variables
    target_x = -1
    target_y = -1
    
    # If contours are found, find the largest one
    if contours and not game_over:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:  # Minimum area threshold
            # Get the centroid of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate target position for drawing the direction indicator
                target_x = cx * GRID_WIDTH // frame.shape[1]
                target_y = cy * GRID_HEIGHT // frame.shape[0]
                
                # Only process movement if cooldown has expired
                if move_cooldown <= 0:
                    # Only allow movement to adjacent cells (no jumping over walls)
                    # Calculate the difference between current and target positions
                    dx = target_x - player_x
                    dy = target_y - player_y
                    
                    # Determine movement direction (one cell at a time)
                    move_x, move_y = 0, 0
                    
                    # Prioritize the larger movement direction
                    if abs(dx) > abs(dy):
                        # Move horizontally
                        move_x = 1 if dx > 0 else -1 if dx < 0 else 0
                    else:
                        # Move vertically
                        move_y = 1 if dy > 0 else -1 if dy < 0 else 0
                    
                    # Calculate new position
                    new_x = player_x + move_x
                    new_y = player_y + move_y
                    
                    # Check if the new position is valid
                    if (0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and 
                        maze[new_y][new_x] != '#'):
                        player_x, player_y = new_x, new_y
                        
                        # Check if player reached the end
                        if maze[player_y][player_x] == 'E':
                            win = True
                            game_over = True
                        
                        # Reset the cooldown timer
                        move_cooldown = MOVE_DELAY
                        
                        # Increment move counter
                        moves_count += 1
    
    # Update cooldown timer
    if move_cooldown > 0:
        move_cooldown -= 1
    
    # Update game time if game is active
    if not game_over and start_time is not None:
        game_time = (pygame.time.get_ticks() - start_time) // 1000  # Convert to seconds
    
    # Draw the maze and player
    screen.blit(background, (0, 0))
    
    # Draw UI panel backgrounds
    right_panel_rect = pygame.Rect(MAZE_LEFT + MAZE_WIDTH + 20, 20, UI_PANEL_WIDTH - 20, WINDOW_HEIGHT - 40)
    draw_panel(screen, right_panel_rect, (40, 40, 60), "Game Controls")
    
    # Draw camera panel
    camera_panel_rect = pygame.Rect(right_panel_rect.left + 10, right_panel_rect.top + 50, 
                                   right_panel_rect.width - 20, 180)
    draw_panel(screen, camera_panel_rect, (30, 30, 50), "Camera View")
    
    # Draw instructions panel
    instructions_panel_rect = pygame.Rect(right_panel_rect.left + 10, camera_panel_rect.bottom + 20, 
                                         right_panel_rect.width - 20, 300)
    draw_panel(screen, instructions_panel_rect, (30, 30, 50), "Instructions")
    
    # Draw maze panel
    maze_panel_rect = pygame.Rect(MAZE_LEFT - 10, MAZE_TOP - 10, 
                                 MAZE_WIDTH + 20, MAZE_HEIGHT + 20)
    draw_panel(screen, maze_panel_rect, (30, 30, 50), "Maze")
    
    # Draw maze
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(MAZE_LEFT + x * CELL_SIZE, MAZE_TOP + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            cell_surface = pygame.Surface((CELL_SIZE, CELL_SIZE))
            
            if maze[y][x] == '#':
                # Draw pixel art grass wall
                draw_pixel_grass(cell_surface, CELL_SIZE)
            elif maze[y][x] == 'E':
                # Draw end point with a pixel art target
                pygame.draw.rect(cell_surface, DARK_GREEN, (0, 0, CELL_SIZE, CELL_SIZE))
                
                # Draw pixel art target/flag
                pixel_size = CELL_SIZE // 8
                
                # Draw flag pole
                for y_pos in range(1, 8):
                    pygame.draw.rect(cell_surface, BROWN, 
                                   (3 * pixel_size, y_pos * pixel_size, 
                                    pixel_size, pixel_size))
                
                # Draw flag
                flag_pattern = [
                    [1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1],
                    [1, 0, 0, 0]
                ]
                
                for fy in range(4):
                    for fx in range(4):
                        if flag_pattern[fy][fx] == 1:
                            pygame.draw.rect(cell_surface, GREEN, 
                                           ((4 + fx) * pixel_size, (1 + fy) * pixel_size, 
                                            pixel_size, pixel_size))
            elif maze[y][x] == 'S':
                # Draw start point with pixel art starting position
                pygame.draw.rect(cell_surface, (50, 50, 70), (0, 0, CELL_SIZE, CELL_SIZE))
                
                # Draw pixel art arrow pointing up
                pixel_size = CELL_SIZE // 8
                arrow_pattern = [
                    [0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0]
                ]
                
                for ay in range(8):
                    for ax in range(6):
                        if arrow_pattern[ay][ax] == 1:
                            pygame.draw.rect(cell_surface, GOLD, 
                                           ((1 + ax) * pixel_size, ay * pixel_size, 
                                            pixel_size, pixel_size))
            else:
                # Draw path with a pixel art pattern
                pygame.draw.rect(cell_surface, (20, 20, 30), (0, 0, CELL_SIZE, CELL_SIZE))
                
                # Create a simple dirt/path texture
                pixel_size = CELL_SIZE // 8
                path_pattern = [
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0]
                ]
                
                for py in range(8):
                    for px in range(8):
                        if path_pattern[py][px] == 1:
                            pygame.draw.rect(cell_surface, (40, 30, 20), 
                                           (px * pixel_size, py * pixel_size, 
                                            pixel_size, pixel_size))
            
            screen.blit(cell_surface, rect)
    
    # Draw path indicator (shows the direction to move)
    if target_x >= 0 and target_y >= 0 and not game_over:
        # Draw a pixel art arrow pointing to the target
        start_pos = (MAZE_LEFT + player_x * CELL_SIZE + CELL_SIZE//2, 
                    MAZE_TOP + player_y * CELL_SIZE + CELL_SIZE//2)
        end_pos = (MAZE_LEFT + target_x * CELL_SIZE + CELL_SIZE//2, 
                  MAZE_TOP + target_y * CELL_SIZE + CELL_SIZE//2)
        
        # Calculate direction vector
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Normalize and scale to get a point a fixed distance away
        length = max(1, (dx**2 + dy**2)**0.5)
        dx = dx / length * 20
        dy = dy / length * 20
        
        # Draw pixel art arrow
        arrow_points = [
            (start_pos[0] + dx, start_pos[1] + dy),
            (start_pos[0] + dy*0.5, start_pos[1] - dx*0.5),
            (start_pos[0] - dy*0.5, start_pos[1] + dx*0.5)
        ]
        
        pygame.draw.polygon(screen, YELLOW, arrow_points)
    
    # Draw player (fox character)
    fox_size = CELL_SIZE - 8  # Slightly smaller than the cell
    fox_surface = draw_pixel_fox(screen, fox_size)
    
    # Add a subtle glow effect behind the fox
    glow_radius = fox_size // 2 + 4
    glow_surface = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
    for r in range(glow_radius, glow_radius-4, -1):
        alpha = 10 + (glow_radius - r) * 20
        if alpha > 100:
            alpha = 100
        pygame.draw.circle(glow_surface, (255, 200, 100, alpha), 
                         (glow_radius, glow_radius), r)
    
    # Position the glow and fox
    player_center = (MAZE_LEFT + player_x * CELL_SIZE + CELL_SIZE//2, 
                    MAZE_TOP + player_y * CELL_SIZE + CELL_SIZE//2)
    
    # Blit the glow onto the screen
    glow_rect = glow_surface.get_rect(center=player_center)
    screen.blit(glow_surface, glow_rect)
    
    # Blit the fox onto the screen
    fox_rect = fox_surface.get_rect(center=player_center)
    screen.blit(fox_surface, fox_rect)
    
    # Display game over or win message
    if game_over:
        # Create a semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        
        # Create a message panel
        message_panel = pygame.Surface((400, 250), pygame.SRCALPHA)
        draw_rounded_rect(message_panel, pygame.Rect(0, 0, 400, 250), (40, 40, 60, 230), radius=20)
        
        # Add a decorative border
        pygame.draw.rect(message_panel, (GOLD if win else RED), pygame.Rect(10, 10, 380, 230), 3, border_radius=15)
        
        # Add the main message
        if win:
            text = title_font.render('You Win!', True, GOLD)
            # Add a trophy icon or stars
            for i in range(5):
                star_pos = (100 + i*50, 100)
                pygame.draw.polygon(message_panel, GOLD, [
                    (star_pos[0], star_pos[1] - 25),
                    (star_pos[0] + 7, star_pos[1] - 10),
                    (star_pos[0] + 23, star_pos[1] - 10),
                    (star_pos[0] + 10, star_pos[1]),
                    (star_pos[0] + 15, star_pos[1] + 15),
                    (star_pos[0], star_pos[1] + 5),
                    (star_pos[0] - 15, star_pos[1] + 15),
                    (star_pos[0] - 10, star_pos[1]),
                    (star_pos[0] - 23, star_pos[1] - 10),
                    (star_pos[0] - 7, star_pos[1] - 10),
                ])
        else:
            text = title_font.render('Game Over', True, RED)
        
        text_rect = text.get_rect(midtop=(200, 30))
        message_panel.blit(text, text_rect)
        
        # Add restart instructions
        font_small = pygame.font.Font(None, 32)
        restart_text = font_small.render('Press R to restart or Q to quit', True, WHITE)
        restart_rect = restart_text.get_rect(midbottom=(200, 220))
        message_panel.blit(restart_text, restart_rect)
        
        # Position and display the message panel
        panel_rect = message_panel.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
        screen.blit(message_panel, panel_rect)
    
    # Display camera view in a small window
    camera_surface = pygame.Surface((UI_PANEL_WIDTH - 40, 150))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (UI_PANEL_WIDTH - 40, 150))
    pygame.surfarray.blit_array(camera_surface, frame_resized.swapaxes(0, 1))
    
    # Add a decorative border to the camera view
    pygame.draw.rect(camera_surface, BLUE, (0, 0, camera_surface.get_width(), camera_surface.get_height()), 2)
    
    # Position the camera view in the camera panel
    camera_pos = (camera_panel_rect.left + 10, camera_panel_rect.top + 40)
    screen.blit(camera_surface, camera_pos)
    
    # Draw a rectangle around the detected object in the camera view
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Scale the rectangle to fit the camera view
            x_scaled = int(x * (UI_PANEL_WIDTH - 40) / frame.shape[1])
            y_scaled = int(y * 150 / frame.shape[0])
            w_scaled = int(w * (UI_PANEL_WIDTH - 40) / frame.shape[1])
            h_scaled = int(h * 150 / frame.shape[0])
            
            # Draw a pulsating rectangle
            pulse = (pygame.time.get_ticks() % 1000) / 1000.0
            thickness = int(2 + pulse * 2)
            pygame.draw.rect(screen, BLUE, 
                           (camera_pos[0] + x_scaled, camera_pos[1] + y_scaled, 
                            w_scaled, h_scaled), thickness)
    
    # Display game stats in the UI panel
    stats_y = instructions_panel_rect.bottom + 20
    time_text = status_font.render(f"Time: {game_time // 60:02d}:{game_time % 60:02d}", True, WHITE)
    moves_text = status_font.render(f"Moves: {moves_count}", True, WHITE)
    
    screen.blit(time_text, (right_panel_rect.left + 20, stats_y))
    screen.blit(moves_text, (right_panel_rect.left + 20, stats_y + 40))
    
    # Display movement cooldown indicator
    if move_cooldown > 0:
        # Create a cooldown bar
        cooldown_rect = pygame.Rect(right_panel_rect.left + 20, WINDOW_HEIGHT - 60, 
                                   right_panel_rect.width - 40, 20)
        
        # Draw background bar
        pygame.draw.rect(screen, (60, 60, 60), cooldown_rect, border_radius=5)
        
        # Draw filled portion based on cooldown
        fill_width = int((MOVE_DELAY - move_cooldown) * cooldown_rect.width / MOVE_DELAY)
        if fill_width > 0:
            fill_rect = pygame.Rect(cooldown_rect.left, cooldown_rect.top, 
                                   fill_width, cooldown_rect.height)
            pygame.draw.rect(screen, BLUE, fill_rect, border_radius=5)
        
        # Draw border
        pygame.draw.rect(screen, WHITE, cooldown_rect, 2, border_radius=5)
        
        # Add label
        cooldown_label = status_font.render("Movement Ready:", True, WHITE)
        screen.blit(cooldown_label, (cooldown_rect.left, cooldown_rect.top - 30))
    else:
        # Show "Ready" message when cooldown is complete
        ready_text = status_font.render("Movement Ready: GO!", True, GREEN)
        screen.blit(ready_text, (right_panel_rect.left + 20, WINDOW_HEIGHT - 90))
    
    # Instructions
    instructions = [
        "- Hold a blue object in front of the camera",
        "- Move the object to guide the fox",
        "- The fox can only move one cell at a time",
        "- You cannot jump over grass walls",
        "- Reach the green flag to win",
        "- Press R to restart",
        "- Press Q to quit"
    ]
    
    # Display instructions in the instructions panel
    for i, line in enumerate(instructions):
        text = instruction_font.render(line, True, WHITE)
        screen.blit(text, (instructions_panel_rect.left + 20, 
                         instructions_panel_rect.top + 40 + i * 30))
    
    # Add game title at the top with pixel art style
    title_text = title_font.render("Pixel Fox Maze Adventure", True, GOLD)
    title_rect = title_text.get_rect(midtop=(WINDOW_WIDTH // 2, 10))
    
    # Add pixel art border around the title
    border_rect = pygame.Rect(title_rect.left - 10, title_rect.top - 5,
                             title_rect.width + 20, title_rect.height + 10)
    
    # Draw pixelated border
    pixel_size = 4
    for x in range(border_rect.left, border_rect.right, pixel_size):
        pygame.draw.rect(screen, GOLD, (x, border_rect.top, pixel_size, pixel_size))
        pygame.draw.rect(screen, GOLD, (x, border_rect.bottom - pixel_size, pixel_size, pixel_size))
    
    for y in range(border_rect.top, border_rect.bottom, pixel_size):
        pygame.draw.rect(screen, GOLD, (border_rect.left, y, pixel_size, pixel_size))
        pygame.draw.rect(screen, GOLD, (border_rect.right - pixel_size, y, pixel_size, pixel_size))
    
    screen.blit(title_text, title_rect)
    
    # Add a status indicator for tracking
    tracking_status = "Tracking: "
    if contours and any(cv2.contourArea(c) > 500 for c in contours):
        tracking_status += "Active"
        status_color = GREEN
    else:
        tracking_status += "Not Detected"
        status_color = RED
    
    status_text = status_font.render(tracking_status, True, status_color)
    screen.blit(status_text, (camera_panel_rect.left + 10, camera_panel_rect.bottom + 10))
    
    pygame.display.flip()
    clock.tick(FPS)

# Clean up
pygame.quit()
cap.release()
cv2.destroyAllWindows()
