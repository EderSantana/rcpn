import sys
import numpy as np
from scipy.misc import imread, imresize
from glob import glob
from natsort import natsorted


def prepare_coil100(size):
    data = np.zeros((100, 72, size, size, 3)).astype('uint8')
    files = glob('./data/coil-100/*.png')
    for i in range(1, 101):
        fimgs = natsorted([f for f in files if "obj{}__".format(i) in f])
        for j, f in enumerate(fimgs):
            print "Loading objects {}, pose {}".format(i, j)
            data[i-1, j] = imresize(imread(f), (size, size))
    return data


if __name__ == "__main__":
    print(sys.argv[1])
    prepare_coil100(int(sys.argv[1]))
