import numpy as np
from tqdm import tqdm
from paint import generate_stroke_sizes, paint, stroke_list
from skimage import transform
import cv2
import matplotlib.pyplot as plt


def paint_image(img, mask, options):
    length, radius = options.length, options.radius

    out = np.full(img.shape, 255)
    diameter = 2 * radius + 1

    stroke_centers = stroke_list(img.shape, radius)

    step_size, strokes = generate_stroke_sizes(mask, length, radius)

    options.quiet = False
    return paint(
        img, out, stroke_centers, strokes, step_size, length, diameter, options
    )


# expects points to be in shape (n, 2)
def get_spaced_centers(good_points, original_points, img_shape, spacing_radius):
    # delete duplicates:
    u, ind = np.unique(good_points, axis=0, return_index=True)
    good_points = u[np.argsort(ind)]

    points_to_include = np.ones((original_points.shape[0] + good_points.shape[0], 2))
    all_points = np.concatenate((good_points, original_points))

    plot = np.zeros(img_shape, dtype=np.uint8)
    plot[good_points[:, 1], good_points[:, 0]] = 1

    # remove points from original_points:
    for i in range(points_to_include.shape[0]):

        point = (int(all_points[i, 1]), int(all_points[i, 0]))

        if (
            point[0] < 0
            or point[0] >= plot.shape[0]
            or point[1] < 0
            or point[1] >= plot.shape[1]
        ):
            points_to_include[i, :] = 0
            continue

        y_window_bounds = (
            max(point[0] - spacing_radius, 0),
            min(point[0] + spacing_radius, plot.shape[0] - 1),
        )
        x_window_bounds = (
            max(point[1] - spacing_radius, 0),
            min(point[1] + spacing_radius, plot.shape[1] - 1),
        )

        points_in_neighborhood = np.sum(
            plot[
                y_window_bounds[0] : y_window_bounds[1],
                x_window_bounds[0] : x_window_bounds[1],
            ]
        )

        # start with assumption that we keep all points
        if i < good_points.shape[0] and points_in_neighborhood > 1:
            points_to_include[i, :] = 0
            plot[point[0], point[1]] = 0
        elif i > good_points.shape[0] and points_in_neighborhood > 0:
            points_to_include[i, :] = 0
        else:
            plot[point[0], point[1]] = 1

    return all_points[points_to_include == 1]


def paint_video(vid, out_path, mask, options):
    """
    I mostly followed this tutorial: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    and these docs: https://docs.opencv.org/3.4/dc/d6b/group__video__track.html

    I used this to download youtube videos: https://github.com/ytdl-org/youtube-dl/blob/master/README.md#readme

    vid should be a filepath
    """
    vidcap = cv2.VideoCapture(vid)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    totalFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    grid = stroke_list(prev_frame.shape, diameter).astype(np.float32)
    p0 = grid

    step_size, strokes = generate_stroke_sizes(mask, length, diameter)

    # every argument except for the first three will be the same for each frame
    options.quiet = True
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

    prev_painted = out
    new_centers = None

    for frame_num in tqdm(range(totalFrames)):
        ret, frame = vidcap.read()
        if not ret:
            print("Exiting video capture")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if new_centers is not None:
            # could just pass in good new
            p0 = get_spaced_centers(new_centers, grid, frame_gray.shape, radius//2 + 1).reshape(-1, 1, 2).astype(np.float32)

        print(f'# points after: {p0.shape[0]}')
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        # use following line to keep all points
        # p1 = p1.reshape(-1,2)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        new_centers = (
            np.concatenate((good_new, good_old)).astype(np.int32).reshape(-1, 2)
        )

        new_centers = new_centers[new_centers[:, 0] >= 0]
        new_centers = new_centers[new_centers[:, 0] < frame_gray.shape[1]]
        new_centers = new_centers[new_centers[:, 1] >= 0]
        new_centers = new_centers[new_centers[:, 1] < frame_gray.shape[0]]

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

    vidcap.release()
    out_vid.release()


def paint_video_naive(vid, out_path, mask, options):
    vidcap = cv2.VideoCapture(vid)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    totalFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = vidcap.read()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # mp4 format
    out_vid = cv2.VideoWriter(
        out_path,
        fourcc,
        vidcap.get(cv2.CAP_PROP_FPS),
        (prev_frame.shape[1], prev_frame.shape[0]),
    )

    length, radius = options.length, options.radius

    diameter = 2 * radius + 1

    stroke_centers = stroke_list(prev_frame.shape, diameter)

    step_size, strokes = generate_stroke_sizes(mask, length, diameter)

    # every argument except for the first three will be the same for each frame
    options.quiet = True
    out = paint(
        prev_frame,
        np.full(prev_frame.shape, 255),
        stroke_centers,
        strokes,
        step_size,
        length,
        diameter,
        options,
    )
    out_vid.write(out)

    for frame_num in tqdm(range(totalFrames)):
        ret, frame = vidcap.read()
        if not ret:
            print("Exiting video capture")
            break

        out = paint(
            frame,
            np.full(frame.shape, 255),
            stroke_centers,
            strokes,
            step_size,
            length,
            diameter,
            options,
        )
        out_vid.write(out)

    vidcap.release()
    out_vid.release()
