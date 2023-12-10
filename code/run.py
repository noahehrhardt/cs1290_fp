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
        good_new = good_new.reshape(-1, 2)
        # good_new = good_new.reshape(-1,2,1)
        # print(f'pass in shape: {good_new.shape}')

        out = np.full(frame.shape, 255)
        out = paint(
            frame,
            out,
            good_new.astype(np.int32),
            strokes,
            step_size,
            length,
            diameter,
            options,
        )
        out_vid.write(out)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    vidcap.release()
    out_vid.release()
