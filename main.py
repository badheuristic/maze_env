import pygame

pygame.init()

TILE_SIZE = 100
WIDTH, HEIGHT = 400, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

grid = [
        [[True, False, True, True], [True, True, False, False]],
        [[True, True, True, True], [False, True, True, False]]
]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    for row_id, row in enumerate(grid):
        for col_id, walls in enumerate(row):
            x = col_id * TILE_SIZE + 50
            y = row_id * TILE_SIZE + 50
            if walls[0]:
                pygame.draw.line(screen, BLACK, (x, y), (x + TILE_SIZE, y), 2)
            if walls[1]:
                pygame.draw.line(screen, BLACK, (x + TILE_SIZE, y), (x + TILE_SIZE, y + TILE_SIZE), 2)
            if walls[2]:
                pygame.draw.line(screen, BLACK, (x + TILE_SIZE, y + TILE_SIZE), (x, y + TILE_SIZE), 2)
            if walls[3]:
                pygame.draw.line(screen, BLACK, (x, y + TILE_SIZE), (x, y), 2)

    pygame.display.flip()

pygame.quit()
