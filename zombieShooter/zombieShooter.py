import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Zombie Shooter")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Player settings
player_img = pygame.image.load('player.png').convert_alpha()  # Load player image
player_x = SCREEN_WIDTH // 2
player_y = SCREEN_HEIGHT // 2
player_speed = 5

# Bullet settings
bullet_img = pygame.Surface((10, 10))
bullet_img.fill(RED)
bullets = []
bullet_speed = 7

# Zombie settings
zombie_img = pygame.image.load('zombie.png').convert_alpha()  # Load zombie image
zombies = []
zombie_speed = 1

# Game clock
clock = pygame.time.Clock()

# Spawn zombies randomly
def spawn_zombie():
    zombie_x = random.choice([0, SCREEN_WIDTH])  # Spawn at left or right edge
    zombie_y = random.randint(0, SCREEN_HEIGHT)  # Random y position
    zombies.append([zombie_x, zombie_y])

# Function to move the player
def move_player(keys, x, y):
    if keys[pygame.K_LEFT]:
        x -= player_speed
    if keys[pygame.K_RIGHT]:
        x += player_speed
    if keys[pygame.K_UP]:
        y -= player_speed
    if keys[pygame.K_DOWN]:
        y += player_speed
    # Keep player within screen bounds
    x = max(0, min(SCREEN_WIDTH - player_img.get_width(), x))
    y = max(0, min(SCREEN_HEIGHT - player_img.get_height(), y))
    return x, y

# Function to shoot bullets
def shoot_bullet(player_x, player_y, mouse_pos):
    bullet_dx = mouse_pos[0] - player_x
    bullet_dy = mouse_pos[1] - player_y
    angle = math.atan2(bullet_dy, bullet_dx)
    velocity_x = math.cos(angle) * bullet_speed
    velocity_y = math.sin(angle) * bullet_speed
    bullets.append([player_x + player_img.get_width() // 2, player_y + player_img.get_height() // 2, velocity_x, velocity_y])

# Function to move zombies towards player
def move_zombies():
    for zombie in zombies:
        zombie_dx = player_x - zombie[0]
        zombie_dy = player_y - zombie[1]
        distance = math.hypot(zombie_dx, zombie_dy)
        if distance > 0:
            zombie[0] += zombie_dx / distance * zombie_speed
            zombie[1] += zombie_dy / distance * zombie_speed

# Function to handle bullet movement and collisions
def handle_bullets():
    for bullet in bullets[:]:  # Copy the list to avoid modifying it while iterating
        bullet[0] += bullet[2]
        bullet[1] += bullet[3]
        if bullet[0] < 0 or bullet[0] > SCREEN_WIDTH or bullet[1] < 0 or bullet[1] > SCREEN_HEIGHT:
            bullets.remove(bullet)
        for zombie in zombies[:]:  # Copy to avoid modifying while iterating
            if pygame.Rect(bullet[0], bullet[1], 10, 10).colliderect(pygame.Rect(zombie[0], zombie[1], zombie_img.get_width(), zombie_img.get_height())):
                bullets.remove(bullet)
                zombies.remove(zombie)
                break

# Main game loop
running = True
spawn_timer = 0

while running:
    screen.fill(BLACK)
    keys = pygame.key.get_pressed()

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            shoot_bullet(player_x, player_y, mouse_pos)

    # Move player
    player_x, player_y = move_player(keys, player_x, player_y)

    # Move bullets
    handle_bullets()

    # Spawn zombies periodically
    spawn_timer += 1
    if spawn_timer > 100:
        spawn_zombie()
        spawn_timer = 0

    # Move zombies towards player
    move_zombies()

    # Draw player
    screen.blit(player_img, (player_x, player_y))

    # Draw bullets
    for bullet in bullets:
        screen.blit(bullet_img, (bullet[0], bullet[1]))

    # Draw zombies
    for zombie in zombies:
        screen.blit(zombie_img, (zombie[0], zombie[1]))

    # Update the screen
    pygame.display.update()

    # Control the game FPS
    clock.tick(60)

# Quit Pygame
pygame.quit()
