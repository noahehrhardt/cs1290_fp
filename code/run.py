import numpy as np
from paint import generate_stroke_sizes, paint, stroke_list
from skimage import transform


def paint_image(img, mask, options):
    length, radius = options.length, options.radius

    out = np.full(img.shape, 255)
    diameter = 2 * radius + 1

    stroke_centers = stroke_list(img.shape, diameter)

    if mask is None:
        mask = np.ones((diameter, diameter))
    else:
        mask = transform.resize(mask, (diameter, diameter))

    step_size, strokes = generate_stroke_sizes(mask, length, diameter)

    return paint(
        img, out, stroke_centers, strokes, step_size, length, diameter, options
    )


def paint_video(vid, mask, options):
    pass
