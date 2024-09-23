import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Invaders")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Load images
player_image = pygame.image.load("player.png").convert_alpha()
alien_image = pygame.image.load("alien.png").convert_alpha()
bullet_image = pygame.image.load("bullet.png").convert_alpha()

# Player class
class Player:
    def __init__(self):
        self.image = player_image
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT - 50))
        self.speed = 5

    def move(self, dx):
        self.rect.x += dx * self.speed
        self.rect.x = max(0, min(WIDTH - self.rect.width, self.rect.x))

    def draw(self, surface):
        surface.blit(self.image, self.rect.topleft)

# Alien class
class Alien:
    def __init__(self, x, y):
        self.image = alien_image
        self.rect = self.image.get_rect(topleft=(x, y))
        self.speed = 1

    def move(self):
        self.rect.y += self.speed

    def draw(self, surface):
        surface.blit(self.image, self.rect.topleft)

# Bullet class
class Bullet:
    def __init__(self, x, y):
        self.image = bullet_image
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = -10

    def move(self):
        self.rect.y += self.speed

    def draw(self, surface):
        surface.blit(self.image, self.rect.topleft)

# Main game function
def main():
    clock = pygame.time.Clock()
    player = Player()
    aliens = []  # Start with an empty list of aliens
    bullets = []
    score = 0

    # Timer for alien generation
    alien_spawn_time = 1000  # milliseconds
    last_spawn_time = pygame.time.get_ticks()
    game_over = False  # Track game state

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player.move(-1)
        if keys[pygame.K_RIGHT]:
            player.move(1)
        if keys[pygame.K_SPACE]:
            if len(bullets) < 5:  # Limit the number of bullets on screen
                bullets.append(Bullet(player.rect.centerx, player.rect.top))

        # Update bullets
        for bullet in bullets[:]:
            bullet.move()
            if bullet.rect.bottom < 0:
                bullets.remove(bullet)

        # Spawn new aliens at intervals
        current_time = pygame.time.get_ticks()
        if current_time - last_spawn_time > alien_spawn_time and not game_over:
            # Randomly place a new alien at the top
            new_alien_x = random.randint(50, WIDTH - 50)
            aliens.append(Alien(new_alien_x, 0))
            last_spawn_time = current_time

        # Update aliens
        for alien in aliens[:]:
            alien.move()
            if alien.rect.top > HEIGHT:
                game_over = True  # Game over if an alien reaches the bottom

        # Check for collisions
        for bullet in bullets[:]:
            for alien in aliens[:]:
                if bullet.rect.colliderect(alien.rect):
                    bullets.remove(bullet)
                    aliens.remove(alien)
                    score += 1
                    break

        # Draw everything
        screen.fill(BLACK)
        player.draw(screen)
        for alien in aliens:
            alien.draw(screen)
        for bullet in bullets:
            bullet.draw(screen)

        # Display score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {score}', True, WHITE)
        screen.blit(score_text, (10, 10))

        # Display game over message
        if game_over:
            game_over_text = font.render('Game Over', True, WHITE)
            screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - 20))
            final_score_text = font.render(f'Final Score: {score}', True, WHITE)
            screen.blit(final_score_text, (WIDTH // 2 - final_score_text.get_width() // 2, HEIGHT // 2 + 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
