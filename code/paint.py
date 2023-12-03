import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from tqdm import tqdm


def stroke_list(img, radius):
    h, w, _ = img.shape

    x_range = np.linspace(0, w - 1, (w - 1) // radius)
    y_range = np.linspace(0, h - 1, (h - 1) // radius)

    x_grid, y_grid = np.meshgrid(x_range, y_range)

    coordinates = np.stack((x_grid, y_grid), axis=-1).reshape((-1, 2)).astype(np.int32)
    np.random.shuffle(coordinates)

    return coordinates


def get_rotated_endpoints(center, length, direction):
    cx, cy = center[0], center[1]

    x1 = cx - length // 2
    x2 = cx + length // 2

    cos = np.cos(np.radians(direction))
    sin = np.sin(np.radians(direction))
    rotation = np.array([[cos, sin], [-sin, cos]])

    end1 = (rotation @ [x1 - cx, 0] + [cx, cy]).astype(np.int32)
    end2 = (rotation @ [x2 - cx, 0] + [cx, cy]).astype(np.int32)

    return end1, end2


def paint_image(img, mask, length, radius, angle, perturb, clip, orient):
    out = np.ones(img.shape)

    stroke_centers = stroke_list(img, radius)

    if mask is None:
        mask = np.ones((2 * radius + 1, 2 * radius + 1))
    else:
        mask = transform.resize(mask, (2 * radius + 1, 2 * radius + 1))

    mask = transform.rotate(mask, angle, True)

    if orient:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gx = cv2.Scharr(img_gray, cv2.CV_32F, dx=1, dy=0)
        gy = cv2.Scharr(img_gray, cv2.CV_32F, dx=0, dy=1)
        directions = np.degrees(np.arctan(gx, gy)) + 90

    for center in tqdm(stroke_centers):
        direction = angle

        if orient:
            direction = directions[center[1], center[0]]

        if perturb:
            direction += int(30 * np.random.rand()) - 15

        end1, end2 = get_rotated_endpoints(center, length, direction)
        diff = end2 - end1

        brush_mask = np.zeros((img.shape[0], img.shape[1]))

        num_steps = max(abs(diff[0]), abs(diff[1]))
        step_size = diff / num_steps
        for i in range(num_steps):
            point = (end1 + i * step_size).astype(np.int32)
            mask_size = mask.shape[0]
            mask_left = point[0] - mask_size // 2
            mask_top = point[1] - mask_size // 2
            mask_region = brush_mask[
                mask_top : mask_top + mask_size, mask_left : mask_left + mask_size
            ]
            if mask_region.shape != mask.shape:
                continue
            mask_region += mask

        brush_mask = np.clip(np.atleast_3d(brush_mask), 0, 1)

        area_under_mask = brush_mask * img

        if np.sum(brush_mask) == 0:
            continue

        color = np.sum(np.reshape(area_under_mask, (-1, 3)), axis=0) / np.sum(
            brush_mask
        )

        if perturb:
            color = np.clip(color + (10 * np.random.rand(3) - 5), 0, 255)

        out = brush_mask * color + (1 - brush_mask) * out

    return np.clip(out, 0, 255).astype(np.uint8)
