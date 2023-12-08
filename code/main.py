import argparse
import sys
import os

import cv2
from paint import paint_image


def main(args):
    in_path = args.input

    if not os.path.exists(in_path):
        print("input img", in_path, "does not exist")
        sys.exit(1)

    in_img = cv2.imread(in_path)

    mask = args.mask
    if mask is not None:
        if not os.path.exists(mask):
            print("brush mask", mask, "does not exist")
            sys.exit(1)

        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    out_img = paint_image(
        in_img,
        mask,
        args.length,
        args.radius,
        args.angle,
        args.perturb,
        args.clip,
        args.orient,
        args.interp,
    )

    if not os.path.isdir("../results"):
        os.mkdir("../results")

    in_name = os.path.splitext(os.path.basename(in_path))[0]
    mask_name = (
        os.path.splitext(os.path.basename(args.mask))[0] if args.mask else "square"
    )

    out_name = f"{in_name}_{mask_name}_{args.length}_{args.radius}_{args.angle}.png"
    cv2.imwrite(
        os.path.join("../results", out_name),
        out_img,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input img/video filepath")
    parser.add_argument("-m", "--mask", help="Brush mask")
    parser.add_argument(
        "-l", "--length", type=int, default=20, help="Brush stroke length"
    )
    parser.add_argument("-r", "--radius", type=int, default=4, help="Brush radius")
    parser.add_argument("-a", "--angle", type=int, default=30, help="Brush angle")
    parser.add_argument(
        "-p",
        "--perturb",
        action="store_false",
        help="Don't randomly perturb stroke colors and angles",
    )
    parser.add_argument(
        "-c", "--clip", action="store_false", help="Clip strokes at edges"
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

    args = parser.parse_args()
    main(args)
