import numpy as np
from paint import generate_stroke_sizes, paint, stroke_list
from skimage import transform
import cv2


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

# currently assuming points are in shape (n, 2)
def get_spaced_centers(good_points, original_points, img_shape, spacing_radius):
    points_to_include = np.ones((original_points.shape[0] + good_points.shape[0], 2))
    all_points = np.concatenate(original_points, good_points)

    plot = np.zeros(img_shape) # make sure this is 1-D
    plot[good_points[1], good_points[0]] = 1

    density_radius = spacing_radius // 2

    # remove points from original_points:
    for i in range(original_points.shape[0]):
    #for i in range(points_to_include.shape[0]):
        if i >= original_points.shape[0]:
            spacing_radius = density_radius

        point = (original_points[i, 1], original_points[i, 0])
        y_window_bounds = (max(point[0] - spacing_radius, 0), min(point[0] + spacing_radius, plot.shape[0]))
        x_window_bounds = (max(point[1] - spacing_radius, 0), min(point[1] + spacing_radius, plot.shape[1]))

        points_in_neighborhood = np.sum(plot[y_window_bounds[0] : y_window_bounds[1], x_window_bounds[0] : x_window_bounds[1]])

        # start with assumption that we keep all points
        if points_in_neighborhood > 0:
            points_to_include[i, :] = 0
        else:
            # only need this if checking for density:
            plot[point[0], point[1]] = 1

    # delete duplicates:
    point_dict = {}
    for i in range(good_points.shape[0]):
        if good_points[i] in point_dict:
            points_to_include[original_points.shape[0] + i, :] = 0
        else:
            point_dict[good_points[i]] = 1
    
    return all_points[points_to_include == 1]


def paint_video(vid, out_path, mask, options):
    """
    I mostly followed this tutorial: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    and these docs: https://docs.opencv.org/3.4/dc/d6b/group__video__track.html

    I used this to download youtube videos: https://github.com/ytdl-org/youtube-dl/blob/master/README.md#readme

    vid should be a filepath
    """
    vidcap = cv2.VideoCapture(vid)

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        # I believe that calcOpticalFlowPyrLK constructs the pyramids itself with maxLevel as the max pyramid level
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    ret, prev_frame = vidcap.read()
    old_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # mp4 format
    out_vid = cv2.VideoWriter(
        out_path,
        fourcc,
        vidcap.get(cv2.CAP_PROP_FPS),
        (prev_frame.shape[1], prev_frame.shape[0]),
    )

    length, radius = options.length, options.radius

    out = np.full(prev_frame.shape, 255)
    diameter = 2 * radius + 1

    p0 = stroke_list(prev_frame.shape, diameter).astype(np.float32)
    print(f"default shape: {p0.shape}, default dtype: {p0.dtype}")
    print(f"max x: {np.max(p0[:, 0])}, max y: {np.max(p0[:, 1])}")

    if mask is None:
        mask = np.ones((diameter, diameter))
    else:
        mask = transform.resize(mask, (diameter, diameter))

    step_size, strokes = generate_stroke_sizes(mask, length, diameter)

    # every argument except for the first three will be the same for each frame
    out = paint(
        prev_frame,
        out,
        p0.astype(np.int32),
        strokes,
        step_size,
        length,
        diameter,
        options,
    )
    out_vid.write(out)

    p0 = p0.reshape(-1, 1, 2)
    print(f"reshaped: {p0.shape}")

    prev_painted = out
    while 1:
        ret, frame = vidcap.read()
        if not ret:
            print("Exiting video capture")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # might need to switch points from (x, y) to (y, x):
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        # print(f'output shape: {p1.shape}')

        # use following line to keep all points
        # p1 = p1.reshape(-1,2)

        # print(f'max x: {np.max(p1[:, 0])}, max y: {np.max(p1[:, 1])}')

        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # good_new = good_new.reshape(-1, 2)
        # good_new = good_new.reshape(-1,2,1)
        # print(f'pass in shape: {good_new.shape}')

        new_centers = (
            np.concatenate((good_new, good_old)).astype(np.int32).reshape(-1, 2)
        )

        # out = np.full(frame.shape, 255)
        out = paint(
            frame,
            prev_painted,
            new_centers,
            strokes,
            step_size,
            length,
            diameter,
            options,
        )
        out_vid.write(out)

        prev_painted = out

        old_gray = frame_gray.copy()
        # p0 = good_new.reshape(-1, 1, 2)

    vidcap.release()
    out_vid.release()
