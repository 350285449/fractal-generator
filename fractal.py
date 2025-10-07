import numpy as np
import pygame
import threading
import time
from numba import njit, prange

np.seterr(over="ignore", invalid="ignore")

WIDTH, HEIGHT = 800, 600
MAX_ITER = 300

surface = None
lock = threading.Lock()
last_input_time = 0
rendering = False


@njit(parallel=True, fastmath=True)
def mandelbrot_jit(width, height, center_x, center_y, scale, max_iter):
    aspect = height / width
    re_min = center_x - scale / 2
    re_max = center_x + scale / 2
    im_min = center_y - (scale * aspect) / 2
    im_max = center_y + (scale * aspect) / 2

    result = np.zeros((height, width), dtype=np.int32)

    for y in prange(height):
        Ci = im_min + (im_max - im_min) * y / height
        for x in range(width):
            Cr = re_min + (re_max - re_min) * x / width
            Zr, Zi = 0.0, 0.0
            count = 0
            while Zr*Zr + Zi*Zi <= 4.0 and count < max_iter:
                Zr, Zi = Zr*Zr - Zi*Zi + Cr, 2*Zr*Zi + Ci
                count += 1
            result[y, x] = count
    return result


def colorize(counts, max_iter):
    norm = counts / max_iter
    norm = np.clip(norm, 0, 1)

    r = (np.sin(6.28 * (norm + 0.0)) * 127 + 128).astype(np.uint8)
    g = (np.sin(6.28 * (norm + 0.33)) * 127 + 128).astype(np.uint8)
    b = (np.sin(6.28 * (norm + 0.66)) * 127 + 128).astype(np.uint8)

    rgb = np.dstack([r, g, b])
    rgb[counts == max_iter] = (0, 0, 0)  # black for inside
    return rgb


def render_fractal(center, scale, max_iter, preview=False):
    global surface, rendering
    if rendering:
        return
    rendering = True
    w, h = (WIDTH//4, HEIGHT//4) if preview else (WIDTH, HEIGHT)
    counts = mandelbrot_jit(w, h, center[0], center[1], scale, max_iter)
    rgb = colorize(counts, max_iter)
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    if preview:
        surf = pygame.transform.scale(surf, (WIDTH, HEIGHT))
    with lock:
        surface = surf
    rendering = False


def start_render(center, scale, max_iter, preview=False):
    threading.Thread(target=render_fractal, args=(center, scale, max_iter, preview), daemon=True).start()


def delayed_full_render(center, scale, max_iter, delay=0.3):
    """Wait a bit after last input, then render full detail"""
    global last_input_time
    this_time = time.time()
    last_input_time = this_time

    def worker():
        time.sleep(delay)
        if last_input_time == this_time:
            start_render(center, scale, max_iter, preview=False)

    threading.Thread(target=worker, daemon=True).start()


def main():
    global surface
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fractal Explorer (Mandelbrot)")

    center = [-0.75, 0.0]
    scale = 3.0
    max_iter = MAX_ITER

    start_render(center, scale, max_iter)

    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)
        changed = False
        step = scale * 0.05

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    center[0] -= step; changed = True
                elif event.key == pygame.K_RIGHT:
                    center[0] += step; changed = True
                elif event.key == pygame.K_UP:
                    center[1] -= step; changed = True
                elif event.key == pygame.K_DOWN:
                    center[1] += step; changed = True
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    scale *= 0.8; max_iter = int(max_iter * 1.05); changed = True
                elif event.key == pygame.K_MINUS:
                    scale /= 0.8; max_iter = max(50, int(max_iter / 1.05)); changed = True

        if changed:
            # Fast preview
            start_render(center, scale, 80, preview=True)
            # Schedule full render after pause
            delayed_full_render(center, scale, max_iter)

        with lock:
            if surface:
                screen.blit(surface, (0, 0))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
