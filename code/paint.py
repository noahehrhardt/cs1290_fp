import numpy as np


def random_stroke_list(img, length, radius):
    h, w, _ = img.shape

    # TODO estimate num strokes from img and brush size
    num_strokes = 10000

    # TODO don't select strokes that go over the edge of image
    rand_c = np.random.randint(0, w, num_strokes)
    rand_r = np.random.randint(0, h, num_strokes)

    return np.stack((rand_r, rand_c), axis=-1)


def paint_image(img, mask, length, radius, angle, clip, orient):
    out = np.zeros(img.shape, dtype=np.uint8)

    stroke_centers = random_stroke_list(img, length, radius)

    for center in stroke_centers:
        cy, cx = center[0], center[1]
        t = max(0, cy - radius)
        b = min(cy + radius, img.shape[0] - 1)
        l = max(0, cx - length // 2)
        r = min(cx + length // 2, img.shape[1] - 1)
        # color = img[cy, cx] # center pix color
        color = np.mean(
            np.reshape(img[t:b, l:r], (-1, 3)), axis=0
        )  # avg color under stroke area
        out[t:b, l:r] = color

    return out
