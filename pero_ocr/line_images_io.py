import os
import cv2


def read_images(folder):
    filenames = [f for f in os.listdir(folder) if f.lower().split('.')[-1] in ['jpg', 'jpeg', 'png']]

    lines = []
    for fn in filenames:
        line_img = cv2.imread(folder + '/' + fn, 1)
        if line_img is None:
            raise ValueError('Error: Could not read image "{}"'.format(fn))
        lines.append(line_img)

    names = ['.'.join(f.split('.')[:-1]) for f in filenames]

    return lines, names
