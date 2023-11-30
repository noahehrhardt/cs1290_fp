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

    in_img = cv2.cvtColor(cv2.imread(in_path), cv2.COLOR_BGR2RGB)

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
        args.clip,
        args.orient,
    )

    if not os.path.isdir("../results"):
        os.mkdir("../results")

    cv2.imwrite(
        os.path.join("../results", "result_" + os.path.basename(in_path)),
        cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input img/video filepath")
    parser.add_argument("-m", "--mask", help="Brush mask")
    parser.add_argument(
        "-l", "--length", type=int, default=30, help="Brush stroke length"
    )
    parser.add_argument("-r", "--radius", type=int, default=5, help="Brush radius")
    parser.add_argument("-a", "--angle", type=int, default=0, help="Brush angle")
    parser.add_argument(
        "-c", "--clip", action="store_true", help="Clip strokes at edges"
    )
    parser.add_argument(
        "-o", "--orient", action="store_true", help="Orient strokes based on gradients"
    )

    args = parser.parse_args()
    main(args)
