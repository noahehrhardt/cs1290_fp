import argparse
import os
import sys

import cv2
from run import paint_image, paint_video, paint_video_naive


def main(args):
    in_path = args.input

    if not os.path.exists(in_path):
        print("input img", in_path, "does not exist")
        sys.exit(1)

    video = in_path.endswith(".mp4") or in_path.endswith(".mov")

    mask = args.mask
    if mask is not None:
        if not os.path.exists(mask):
            print("brush mask", mask, "does not exist")
            sys.exit(1)

        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    in_name = os.path.splitext(os.path.basename(in_path))[0]
    mask_name = (
        os.path.splitext(os.path.basename(args.mask))[0] if args.mask else "square"
    )

    print("Rendering", in_name)

    config = f"{'p' if args.perturb else ''}{'c' if args.clip else ''}{'o' if args.orient else ''}{'i' if args.interp else ''}{'f' if args.flow and video else ''}"
    angle = "" if args.orient else f"_a{args.angle}"
    out_name = f"{in_name}_{mask_name}_l{args.length}_r{args.radius}{angle}{f'_{config}' if config != '' else ''}"

    if not os.path.isdir("../results"):
        os.mkdir("../results")

    if not video:
        in_img = cv2.imread(in_path)
        out_img = paint_image(in_img, mask, args)

        out_name += ".png"
        cv2.imwrite(
            os.path.join("../results", out_name),
            out_img,
        )
    else:
        out_name += ".mp4"
        out_path = os.path.join("../results", out_name)

        if args.flow:
            paint_video(in_path, out_path, mask, args)
        else:
            paint_video_naive(in_path, out_path, mask, args)

    print("Wrote output to", out_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input img/video filepath")
    parser.add_argument("-m", "--mask", help="Brush mask filepath")
    parser.add_argument(
        "-l", "--length", type=int, default=20, help="Brush stroke length (int)"
    )
    parser.add_argument(
        "-r", "--radius", type=int, default=4, help="Brush radius (int)"
    )
    parser.add_argument("-a", "--angle", type=int, default=30, help="Brush angle (int)")
    parser.add_argument(
        "-p",
        "--perturb",
        action="store_false",
        help="Don't randomly perturb stroke colors and angles",
    )
    parser.add_argument(
        "-c", "--clip", action="store_false", help="Don't clip strokes at edges"
    )
    parser.add_argument(
        "-o",
        "--orient",
        action="store_false",
        help="Don't orient strokes based on gradients",
    )
    parser.add_argument(
        "-i",
        "--interp",
        action="store_false",
        help="Don't interpolate stroke gradient directions",
    )
    parser.add_argument(
        "-v",
        "--view",
        action="store_true",
        help="Show view of stroke placement in real time",
    )
    parser.add_argument(
        "-f",
        "--flow",
        action="store_false",
        help="Don't use optical flow to optimize painting video frames",
    )

    args = parser.parse_args()
    main(args)
