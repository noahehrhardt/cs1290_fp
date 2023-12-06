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

    return step_size, strokes


def get_canny_edges(img):
    img_gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 5)
    # https://stackoverflow.com/questions/21324950/how-can-i-select-the-best-set-of-parameters-in-the-canny-edge-detection-algorith
    high, thresh_im = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    low = 0.5 * high
    edges = cv2.Canny(img_gray, low, high)
    return edges


def dist_to_edge(center, length, direction, edges):
    line_x = np.arange(0, length) - length // 2
    line_y = line_x * np.sin(np.radians(direction))
    line_x = np.int32(line_x + center[0])
    line_y = np.int32(line_y + center[1])
    line = np.zeros(edges.shape)
    line[line_y, line_x] = 1
    hit_edges = np.argwhere((line * edges) > 0)
    if hit_edges.shape[0] > 0:
        return np.sqrt(
            np.min(
                np.sum(
                    np.square(hit_edges - [center[1], center[0]]),
                    axis=-1,
                )
            )
        )
    return -1


def paint_image(img, mask, length, radius, angle, perturb, clip, orient):
    out = np.full(img.shape, 255)

    stroke_centers = stroke_list(img, radius)

    if mask is None:
        mask = np.ones((2 * radius + 1, 2 * radius + 1))
    else:
        mask = transform.resize(mask, (2 * radius + 1, 2 * radius + 1))

    step_size, strokes = generate_stroke_sizes(mask, length, radius)

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
        edges = get_canny_edges(img)

    for center in tqdm(stroke_centers):
        # just skip anything close to the edges for now
        if (
            center[1] < length * 1.5
            or center[1] > img.shape[0] - length * 1.5
            or center[0] < length * 1.5
            or center[0] > img.shape[1] - length * 1.5
        ):
            continue

        direction = angle
        if orient:
            direction = directions[center[1], center[0]]
        if perturb:
            direction += int(30 * np.random.rand()) - 15

        stroke_len = length
        if clip:
            closest_edge_dist = dist_to_edge(center, length, direction, edges)
            if closest_edge_dist > -1:
                if closest_edge_dist < step_size:
                    stroke_len = step_size
                else:
                    stroke_len = int(int(closest_edge_dist / step_size) * step_size)

        rotated_stroke = transform.rotate(strokes[stroke_len], direction, True)

        height, width = rotated_stroke.shape
        mask_left = center[0] - width // 2
        mask_top = center[1] - height // 2

        rotated_stroke = np.atleast_3d(rotated_stroke)

        color = blurred[center[1], center[0]]

        if perturb:
            color = np.clip(color + (20 * np.random.rand() - 10), 0, 255)

        out[mask_top : mask_top + height, mask_left : mask_left + width] = (
            rotated_stroke * color
            + (1 - rotated_stroke)
            * out[mask_top : mask_top + height, mask_left : mask_left + width]
        )

        # cv2.imshow("painting", np.clip(out, 0, 255).astype(np.uint8))
        # cv2.waitKey(1)

    return np.clip(out, 0, 255).astype(np.uint8)
