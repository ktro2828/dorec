#!/usr/bin/env python

import cv2

from dorec.core.utils import TokenParser


def main():
    root = [
        "/media/leus/cae08de1-58b1-4e2f-9b2d-5890104f960d/Deformable-Object-Recognition/data/20211119/double/HalfShirt",
        "/media/leus/cae08de1-58b1-4e2f-9b2d-5890104f960d/Deformable-Object-Recognition/data/20211119/double/HalfShirt0",
        "/media/leus/cae08de1-58b1-4e2f-9b2d-5890104f960d/Deformable-Object-Recognition/data/20211119/double/HalfShirt1",
        "/media/leus/cae08de1-58b1-4e2f-9b2d-5890104f960d/Deformable-Object-Recognition/data/20211119/double/Towel",
        "/media/leus/cae08de1-58b1-4e2f-9b2d-5890104f960d/Deformable-Object-Recognition/data/20211119/double/Towel0",
        "/media/leus/cae08de1-58b1-4e2f-9b2d-5890104f960d/Deformable-Object-Recognition/data/20211119/single/HalfShirt",
        "/media/leus/cae08de1-58b1-4e2f-9b2d-5890104f960d/Deformable-Object-Recognition/data/20211119/single/Towel",
        "/media/leus/cae08de1-58b1-4e2f-9b2d-5890104f960d/Deformable-Object-Recognition/data/20211121/double/HalfShirt",
        "/media/leus/cae08de1-58b1-4e2f-9b2d-5890104f960d/Deformable-Object-Recognition/data/20211121/double/HalfShirt0",
        "/media/leus/cae08de1-58b1-4e2f-9b2d-5890104f960d/Deformable-Object-Recognition/data/20211121/double/Towel",
        "/media/leus/cae08de1-58b1-4e2f-9b2d-5890104f960d/Deformable-Object-Recognition/data/20211121/double/Towel0"
    ]

    blue = 0.0
    green = 0.0
    red = 0.0
    black = 0.0
    for r in root:
        tp = TokenParser(r)
        num_data = len(tp.tokens_list)
        for idx in range(num_data):
            mask_path = tp.getFilepath(idx, "mask")
            mask = cv2.imread(mask_path)
            blue += float((mask[:, :, 0] > 0).sum())
            green += float((mask[:, :, 1] > 0).sum())
            red += float((mask[:, :, 2] > 0).sum())
            black += float((mask == 0).sum())

    total = blue + green + red + black
    print("Blue: ", blue / total)
    print("Green: ", green / total)
    print("Red: ", red / total)
    print("Black: ", black / total)


if __name__ == "__main__":
    main()
