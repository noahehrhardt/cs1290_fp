import numpy as np
from paint import generate_stroke_sizes, paint, stroke_list
from skimage import transform
import cv2


def paint_image(img, mask, options):
    length, radius = options.length, options.radius

    out = np.full(img.shape, 255)
    diameter = 2 * radius + 1

    stroke_centers = stroke_list(img.shape, radius)

    step_size, strokes = generate_stroke_sizes(mask, length, radius)

    return paint(
        img, out, stroke_centers, strokes, step_size, length, diameter, options
    )


# currently assuming points are in shape (n, 2)
def get_spaced_centers(good_points, original_points, img_shape, spacing_radius=1):
    # delete duplicates:
    u, ind = np.unique(good_points, axis=0, return_index=True)
    good_points = u[np.argsort(ind)]

    points_to_include = np.ones((original_points.shape[0] + good_points.shape[0], 2))
    all_points = np.concatenate((good_points, original_points))
    
    print(f'# points before: {all_points.shape[0]}')

    plot = np.zeros(img_shape, dtype=np.uint8) # make sure this is 1-D
    plot[good_points[:, 1], good_points[:, 0]] = 1

    # remove points from original_points:
    #for i in range(original_points.shape[0]):
    for i in reversed(range(points_to_include.shape[0])):

        point = (int(all_points[i, 1]), int(all_points[i, 0]))

        if point[0] < 0 or point[0] >= plot.shape[0] or point[1] < 0 or point[1] >= plot.shape[1]:
            points_to_include[i, :] = 0
            continue

        y_window_bounds = (max(point[0] - spacing_radius, 0), min(point[0] + spacing_radius, plot.shape[0] - 1))
        x_window_bounds = (max(point[1] - spacing_radius, 0), min(point[1] + spacing_radius, plot.shape[1] - 1))

        points_in_neighborhood = np.sum(
            plot[
                y_window_bounds[0] : y_window_bounds[1],
                x_window_bounds[0] : x_window_bounds[1],
            ]
        )

        # start with assumption that we keep all points
        if points_in_neighborhood > 0:
            points_to_include[i, :] = 0
        else:
            # only need this if checking for density:
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

    while 1:
        ret, frame = vidcap.read()
        if not ret:
            print("Exiting video capture")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if new_centers is not None:
            p0 = get_spaced_centers(new_centers, grid, frame_gray.shape, 1).reshape(-1, 1, 2).astype(np.float32)
        print(f'# points after: {p0.shape[0]}')
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
        # p0 = good_new.reshape(-1, 1, 2)

    vidcap.release()
    out_vid.release()


def optical_flow(vid, out_path):
    cap = cv2.VideoCapture(vid)

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # mp4 format
    out_vid = cv2.VideoWriter(
        out_path,
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (old_frame.shape[1], old_frame.shape[0]),
    )

    while(1):
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            a = int(a)
            b = int(b)
            c = int(c)
            d = int(d)
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        out_vid.write(img)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()