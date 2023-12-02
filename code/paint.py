import numpy as np
from skimage import transform
import cv2


def stroke_list(img):
    h, w, _ = img.shape

    x_range = np.linspace(0, w - 1, (w - 1) // 2)
    y_range = np.linspace(0, h - 1, (h - 1) // 2)

    x_grid, y_grid = np.meshgrid(x_range, y_range)

    coordinates = np.stack((x_grid, y_grid), axis=-1).reshape((-1, 2)).astype(np.int32)
    np.random.shuffle(coordinates)

    return coordinates


def paint_image(img, mask, length, radius, angle, perturb, clip, orient):
    out = np.zeros(img.shape, dtype=np.uint8)

    if orient:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gx = cv2.Scharr(img_gray, cv2.CV_32F, dx=1, dy=0)
        gy = cv2.Scharr(img_gray, cv2.CV_32F, dx=0, dy=1)
        directions = np.degrees(np.arctan(gx, gy)) + 90

    stroke_centers = stroke_list(img)

    if mask is None:
        mask = np.ones((2 * radius + 1, 2 * radius + 1))
    else:
        mask = transform.resize(mask, (2 * radius + 1, 2 * radius + 1))

    mask = transform.rotate(mask, angle, True)

    for center in stroke_centers:
        direction = angle
        cy, cx = center[0], center[1]

        if orient:
            # TODO: figure out angle based on gradient
            angle = directions[cy, cx]

        x1 = max(0, cx - length // 2)
        x2 = min(cx + length // 2, img.shape[1] - 1)

        cos = np.cos(np.radians(direction))
        sin = np.sin(np.radians(direction))
        rotation = np.array([[cos, -sin], [sin, cos]])

        end1 = rotation @ [x1 - cx, 0]
        end2 = rotation @ [x2 - cx, 0]
        end1 += [cx, cy]
        end2 += [cx, cy]
        end1 = end1.astype(np.int32)
        end2 = end2.astype(np.int32)
        
        diff = end2 - end1
        num_steps = max(diff[0], diff[1])
        step_size = diff / num_steps
        for i in range(num_steps):
            point = (end1 + i * step_size).astype(np.int32)

        # color = np.mean(np.reshape(img[t:b, l:r], (-1, 3)), axis=0)

        # if perturb:
        #     color = np.clip(color + (10 * np.random.rand(3) - 5), 0, 255)

        # if clip:
        #     pass
        # else:
        #     out[t:b, l:r] = color

    return out
