import cv2
import numpy as np
from scipy import interpolate
from skimage import transform
from tqdm import tqdm
import matplotlib.pyplot as plt


def stroke_list(shape, radius):
    h, w, _ = shape
    diameter = 2 * radius + 1

    x_range = np.linspace(diameter, w - diameter - 1, (w - 2 * diameter - 1) // radius)
    y_range = np.linspace(diameter, h - diameter - 1, (h - 2 * diameter - 1) // radius)

    x_grid, y_grid = np.meshgrid(x_range, y_range)

    coordinates = np.stack((x_grid, y_grid), axis=-1).reshape((-1, 2)).astype(np.int32)
    np.random.shuffle(coordinates)

    return coordinates


def generate_stroke_sizes(mask, length, radius):
    strokes = {}
    step_size = 1
    diameter = 2 * radius + 1

    if mask is not None:
        resized = transform.rescale(mask, diameter / mask.shape[0])

    if length > 20:
        step_size = length / 20

    size = step_size
    while size <= length:
        stroke_len = int(size)
        if mask is not None:
            strokes[stroke_len] = np.hstack(
                (resized[:, : radius + stroke_len], resized[:, -radius:])
            )
        else:
            strokes[stroke_len] = np.ones((diameter, stroke_len + diameter))

        size += step_size

    return step_size, strokes


def get_canny_edges(img, diameter, clip):
    if clip:
        img_gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 5)
        # https://stackoverflow.com/questions/21324950/how-can-i-select-the-best-set-of-parameters-in-the-canny-edge-detection-algorith
        high, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low = 0.5 * high
        edges = cv2.Canny(img_gray, low, high)

    else:
        edges = np.zeros((img.shape[0], img.shape[1]))

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


def gradient_directions(img, interp):
    img_gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (11, 11), 11)
    gx = cv2.GaussianBlur(cv2.Scharr(img_gray, cv2.CV_32F, dx=1, dy=0), (5, 5), 5)
    gy = cv2.GaussianBlur(cv2.Scharr(img_gray, cv2.CV_32F, dx=0, dy=1), (5, 5), 5)

    if interp:
        threshold = max(np.mean(np.abs(gx)), np.mean(np.abs(gy)))

        gx[np.abs(gx) < threshold] = 0
        gy[np.abs(gy) < threshold] = 0

        x_important = np.nonzero(gx)
        y_important = np.nonzero(gy)

        x_needed = np.nonzero(gx == 0)
        y_needed = np.nonzero(gy == 0)

        x_vals = gx[x_important]
        y_vals = gy[y_important]

        x_interp = interpolate.griddata(
            x_important,
            x_vals,
            x_needed,
            method="linear",
            fill_value=0,
        )
        y_interp = interpolate.griddata(
            y_important,
            y_vals,
            y_needed,
            method="linear",
            fill_value=0,
        )

        gx[x_needed] = x_interp
        gy[y_needed] = y_interp

    directions = np.degrees(np.arctan2(gx, gy))

    return directions


def paint(img, out, stroke_centers, strokes, step_size, length, diameter, options):
    blurred = cv2.GaussianBlur(img, (5, 5), 5)

    if options.orient:
        directions = gradient_directions(img, options.interp)

    edges = get_canny_edges(img, diameter, options.clip)

    enumerable = stroke_centers
    if not options.quiet:
        enumerable = tqdm(stroke_centers)
    for i, center in enumerate(enumerable):
        if (
            center[0] < diameter
            or center[1] < diameter
            or center[0] > img.shape[1] - diameter
            or center[1] > img.shape[0] - diameter
        ):
            continue

        direction = options.angle
        if options.orient:
            direction = directions[center[1], center[0]]
        if options.perturb:
            direction += int(30 * np.random.rand()) - 15

        stroke_len = length
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
        mask_right = mask_left + width
        mask_bottom = mask_top + height

        if (
            mask_left < 0
            or mask_right >= img.shape[1]
            or mask_top < 0
            or mask_bottom >= img.shape[0]
        ):
            continue

        rotated_stroke = np.atleast_3d(rotated_stroke)

        color = blurred[center[1], center[0]]

        if options.perturb:
            color = np.clip(color + (20 * np.random.rand() - 10), 0, 255)

        out[mask_top:mask_bottom, mask_left:mask_right] = (
            rotated_stroke * color
            + (1 - rotated_stroke) * out[mask_top:mask_bottom, mask_left:mask_right]
        )

        if options.view and i % 1000 == 0:
            cv2.imshow("painting", np.clip(out, 0, 255).astype(np.uint8))
            cv2.waitKey(1)

    return np.clip(out, 0, 255).astype(np.uint8)
