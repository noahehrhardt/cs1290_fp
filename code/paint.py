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


def generate_stroke_sizes(mask, length, radius):
    strokes = {}
    step_size = 1

    if length > 10:
        step_size = length / 10

    side_len = 2 * radius + 1

    size = step_size
    while size <= length:
        stroke_len = int(size)
        stroke_mask = np.zeros((2 * radius + 1, stroke_len + 2 * radius))
        for i in range(stroke_len):
            stroke_mask[0:side_len, i : i + side_len] += mask
        strokes[stroke_len] = np.clip(stroke_mask, 0, 1)
        size += step_size

    return strokes


def paint_image(img, mask, length, radius, angle, perturb, clip, orient):
    out = np.full(img.shape, 255)

    stroke_centers = stroke_list(img, radius)

    if mask is None:
        mask = np.ones((2 * radius + 1, 2 * radius + 1))
    else:
        mask = transform.resize(mask, (2 * radius + 1, 2 * radius + 1))

    strokes = generate_stroke_sizes(mask, length, radius)

    blurred = cv2.GaussianBlur(img, (5, 5), 5)

    if orient:
        img_gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (11, 11), 11)
        gx = cv2.Scharr(img_gray, cv2.CV_32F, dx=1, dy=0)
        gy = cv2.Scharr(img_gray, cv2.CV_32F, dx=0, dy=1)
        directions = np.degrees(np.arctan2(gx, gy))
        directions = np.where(
            np.any((np.abs(gx) > 10, np.abs(gy) > 10)), directions, angle
        )

    if clip:
        img_gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 5)
        # https://stackoverflow.com/questions/21324950/how-can-i-select-the-best-set-of-parameters-in-the-canny-edge-detection-algorith
        high, thresh_im = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        low = 0.5 * high
        edges = cv2.Canny(img_gray, low, high)

    for center in tqdm(stroke_centers):
        if (
            center[1] < length
            or center[1] > img.shape[0] - length - 1
            or center[0] < length
            or center[0] > img.shape[1] - length
        ):
            continue

        direction = angle

        if orient:
            direction = directions[center[1], center[0]]

        if perturb:
            direction += int(30 * np.random.rand()) - 15

        brush_mask = np.zeros((img.shape[0], img.shape[1]))

        rotated_stroke = transform.rotate(strokes[length], direction, True)
        height, width = rotated_stroke.shape
        mask_left = center[0] - width // 2
        mask_top = center[1] - height // 2
        brush_mask[
            mask_top : mask_top + height, mask_left : mask_left + width
        ] = rotated_stroke

        brush_mask = np.atleast_3d(brush_mask)

        color = blurred[center[1], center[0]]

        if perturb:
            color = np.clip(color + (20 * np.random.rand() - 10), 0, 255)

        out = brush_mask * color + (1 - brush_mask) * out

        # cv2.imshow("painting", np.clip(out, 0, 255).astype(np.uint8))
        # cv2.waitKey(1)

    return np.clip(out, 0, 255).astype(np.uint8)
