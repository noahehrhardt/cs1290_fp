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


def paint_video(vid, mask, options):
    """
    I mostly followed this tutorial: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    and these docs: https://docs.opencv.org/3.4/dc/d6b/group__video__track.html
    """
    vidcap = cv2.VideoCapture(vid)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                     # I believe that calcOpticalFlowPyrLK constructs the pyramids itself with maxLevel as the max pyramid level
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    ret, prev_frame = vidcap.read()
    old_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    length, radius = options.length, options.radius

    out = np.full(prev_frame.shape, 255)
    out_vid = []
    diameter = 2 * radius + 1

    p0 = stroke_list(prev_frame.shape, diameter)

    if mask is None:
        mask = np.ones((diameter, diameter))
    else:
        mask = transform.resize(mask, (diameter, diameter))

    step_size, strokes = generate_stroke_sizes(mask, length, diameter)

    # every argument except for the first three will be the same for each frame
    out = paint(prev_frame, out, p0, strokes, step_size, length, diameter, options)
    out_vid.append(out)


    while(1):
        ret, frame = vidcap.read()
        if not ret:
            print("Exiting video capture")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # might need to switch points from (x, y) to (y, x):
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st==1]

        out = np.full(frame.shape, 255)
        out = paint(frame, out, good_new, strokes, step_size, length, diameter, options)
        out_vid.append(out)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    
    # return list of painted frames:
    return out_vid