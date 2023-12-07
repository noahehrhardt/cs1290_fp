import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from tqdm import tqdm
from scipy import interpolate


def stroke_list(shape, radius):
    h, w, _ = shape
    diameter = 2 * radius + 1

    x_range = np.linspace(diameter, w - diameter - 1, (w - 2 * diameter - 1) // 2)
    y_range = np.linspace(diameter, h - diameter - 1, (h - 2 * diameter - 1) // 2)

    x_grid, y_grid = np.meshgrid(x_range, y_range)

    coordinates = np.stack((x_grid, y_grid), axis=-1).reshape((-1, 2)).astype(np.int32)
    np.random.shuffle(coordinates)

    return coordinates


def generate_stroke_sizes(mask, length, radius):
    strokes = {}
    step_size = 1

    if length > 20:
        step_size = length / 20

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


def get_canny_edges(img, radius):
    img_gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 5)
    # https://stackoverflow.com/questions/21324950/how-can-i-select-the-best-set-of-parameters-in-the-canny-edge-detection-algorith
    high, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = 0.5 * high
    edges = cv2.Canny(img_gray, low, high)
    diameter = 2 * radius + 1
    edges[diameter, :] = 255
    edges[-diameter, :] = 255
    edges[:, diameter] = 255
    edges[:, -diameter] = 255
    return edges


def dist_to_edge(center, length, direction, edges):
    line_x = np.arange(0, length) - length // 2
    line_y = line_x * np.sin(np.radians(direction))
    line_x = np.int32(line_x + center[0])
    line_y = np.int32(line_y + center[1])
    valid_coords = (
        (line_x >= 0)
        & (line_x < edges.shape[1])
        & (line_y >= 0)
        & (line_y < edges.shape[0])
    )

    line_x = line_x[valid_coords]
    line_y = line_y[valid_coords]

    hit_idx = np.argwhere(edges[line_y, line_x] > 0)
    hit_edges = np.concatenate((line_x[hit_idx], line_y[hit_idx]), axis=1)

    if hit_edges.shape[0] > 0:
        return np.sqrt(
            np.min(
                np.sum(
                    np.square(hit_edges - center),
                    axis=-1,
                )
            )
        )
    return -1


def paint_image(img, mask, length, radius, angle, perturb, clip, orient):
    out = np.full(img.shape, 255)
    diameter = 2 * radius + 1

    stroke_centers = stroke_list(img.shape, radius)

    if mask is None:
        mask = np.ones((diameter, diameter))
    else:
        mask = transform.resize(mask, (diameter, diameter))

    step_size, strokes = generate_stroke_sizes(mask, length, radius)

    blurred = cv2.GaussianBlur(img, (5, 5), 5)

    if orient:
        img_gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (11, 11), 11)
        gx = cv2.Scharr(img_gray, cv2.CV_32F, dx=1, dy=0)
        gy = cv2.Scharr(img_gray, cv2.CV_32F, dx=0, dy=1)

        gx[abs(gx) < 50] = 0
        gy[abs(gy) < 50] = 0
        # maybe check that both less than 10

        x_important = np.nonzero(gx)
        y_important = np.nonzero(gy)

        x_vals = gx[x_important[0], x_important[1]]
        y_vals = gy[y_important[0], y_important[1]]

        plt.imshow(gx, cmap='gray')
        plt.show()

        gx_interp = interpolate.griddata(x_important, x_vals, (stroke_centers[:, 1], stroke_centers[:, 0]), method='cubic', fill_value=0)
        gy_interp = interpolate.griddata(y_important, y_vals, (stroke_centers[:, 1], stroke_centers[:, 0]), method='cubic', fill_value=0)

        gx[stroke_centers[:, 1], stroke_centers[:, 0]] = gx_interp
        gy[stroke_centers[:, 1], stroke_centers[:, 0]] = gy_interp

        plt.imshow(gx, cmap='gray')
        plt.show()

        #directions = np.degrees(np.arctan2(gx, gy))
        directions = np.degrees(np.arctan2(gx, gy))

        '''directions = np.where(
            np.any((np.abs(gx) > 10, np.abs(gy) > 10)), directions, angle
        )'''

    if clip:
        edges = get_canny_edges(img, radius)

    for center in tqdm(stroke_centers):
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
                    stroke_len = int(step_size)
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
