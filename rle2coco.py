from pycocotools import mask
import numpy as np
import cv2
import os


def main():
    content = np.loadtxt('train.csv', str, delimiter=',', skiprows=1)
    # content[0, 1:]
    dic = dict()
    dic[content[0, 1]] = content[0, 2]

    m = mask.decode(dic)
    print(m)


if __name__ == "__main__":
    main()
