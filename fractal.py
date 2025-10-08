import numpy as np
import pygame
import threading
import time
from numba import njit, prange

# =============== CONFIGURATION ===============
WIDTH, HEIGHT = 800, 600
INITIAL_SCALE = 3.0
MAX_ITER = 300
PREVIEW_ITER = 80
ZOOM_FACTOR = 0.8
MOVE_FACTOR = 0.05
PREVIEW_DELAY = 0.25
FPS = 60
# =============================================

np.seterr(over="ignore", invalid="ignore")

# Shared global state
surface = None
surface_lock = threading.Lock()
render_id = 0  # used to cancel old renders


@njit(parallel=True, fastmath=True)
def mandelbrot_jit(width, height, cx, cy, scale, max_iter):
    aspect = height / width
    re_min, re_max = cx - scale / 2, cx + scale / 2
    im_min, im_max = cy - (scale * aspect) / 2, cy + (scale * aspect) / 2

    result = np.zeros((height, width), dtype=np.int32)
    for y in prange(height):
        ci = im_min + (im_max - im_min) * y / height
        for x in range(width):
            cr = re_min + (re_max - re_min) * x / width
            zr = zi = 0.0
            count = 0
            while zr * zr + zi * zi <= 4.0 and count < max_iter:
                zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
                count += 1
            result[y, x] = count
    return result


def colorize(counts, max_iter):
    norm = np.clip(counts / max_iter, 0, 1)
    r = (np.sin(6.283 * (norm + 0.0)) * 127 + 128).astype(np.uint8)
    g = (np.sin(6.283 * (norm + 0.33)) * 127 + 128).astype(np.uint8)
    b = (np.sin(6.283 * (norm + 0.66)) * 127 + 128).astype(np.uint8)
    rgb = np.dstack([r, g, b])
    rgb[counts == max_iter] = (0, 0, 0)
    return rgb


def render_fractal(center, scale, max_iter, preview, job_id):
    """Background worker that computes and safely sets the current surface."""
    w, h = (WIDTH // 4, HEIGHT // 4) if preview else (WIDTH, HEIGHT)
    counts = mandelbrot_jit(w, h, center[0], center[1], scale, max_iter)
    rgb = colorize(counts, max_iter)
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    if preview:
        surf = pygame.transform.smoothscale(surf, (WIDTH, HEIGHT))

    # Check if job is still valid (no newer render started)
    if job_id != render_id:
        return
    with surface_lock:
        global surface
        surface = surf


def start_render(center, scale, max_iter, preview=False):
    """Start a new render in background, canceling older ones."""
    global render_id
    render_id += 1
    job_id = render_id
    threading.Thread(
        target=render_fractal, args=(center, scale, max_iter, preview, job_id), daemon=True
    ).start()


def delayed_full_render(center, scale, max_iter, delay=PREVIEW_DELAY):
    """Schedule full-quality render after user stops moving."""
    this_id = render_id

    def worker():
        time.sleep(delay)
        if this_id == render_id:
            start_render(center, scale, max_iter, preview=False)

    threading.Thread(target=worker, daemon=True).start()


def main():
    global surface
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mandelbrot Explorer (Numba + Threads)")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    center = [-0.75, 0.0]
    scale = INITIAL_SCALE
    max_iter = MAX_ITER

    start_render(center, scale, max_iter)

    running = True
    show_fps = True

    while running:
        dt = clock.tick(FPS) / 1000.0
        changed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                step = scale * MOVE_FACTOR
                if event.key == pygame.K_LEFT:
                    center[0] -= step; changed = True
                elif event.key == pygame.K_RIGHT:
                    center[0] += step; changed = True
                elif event.key == pygame.K_UP:
                    center[1] -= step; changed = True
                elif event.key == pygame.K_DOWN:
                    center[1] += step; changed = True
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    scale *= ZOOM_FACTOR
                    max_iter = int(max_iter * 1.05)
                    changed = True
                elif event.key == pygame.K_MINUS:
                    scale /= ZOOM_FACTOR
                    max_iter = max(50, int(max_iter / 1.05))
                    changed = True
                elif event.key == pygame.K_f:
                    show_fps = not show_fps

        if changed:
            start_render(center, scale, PREVIEW_ITER, preview=True)
            delayed_full_render(center, scale, max_iter)

        with surface_lock:
            if surface:
                screen.blit(surface, (0, 0))

        if show_fps:
            fps_text = font.render(f"{clock.get_fps():.1f} FPS", True, (255, 255, 255))
            screen.blit(fps_text, (10, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
