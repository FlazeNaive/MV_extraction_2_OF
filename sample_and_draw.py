import matplotlib.pyplot as plt
import flowiz as fz
from scipy.ndimage import zoom
import cv2
import numpy as np
import os

# ================ read *.flo ================
def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """

    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)[0]
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

def vis(a, name="default"):
    print("NAME ", name)
    img = fz.convert_from_flow(a)
    dict_order = ["out", "gt", "raft"]
    for (i, dic) in enumerate(dict_order):
        shape = np.shape(img)
        print(shape)
        h = int(shape[0]/3)
        print(h)
        cur = img[i*h: (i+1)*h, :, :]

        OUTPATH = "tmp_visual/"
        if not os.path.exists(OUTPATH):
            os.makedirs(OUTPATH)
        FILENAME = dict_order[i] + "-" + name
        print(FILENAME)
        cv2.imwrite(OUTPATH + FILENAME, cur)

def paint(scene, fr_num):
    # print("Loading OUTPUT...")
    OUT_PATH = "data-output/" + scene + "/mv-flo/flow-" + str(fr_num) + ".flo"
    out = read_flo_file(OUT_PATH)
    # print("Loading GT...")
    GT_PATH = "data-gtflow/" + scene + "/frame_" + str(fr_num).zfill(4) + ".flo"
    gt = read_flo_file(GT_PATH)
    # print("Loading RAFT...")
    RAFT_PATH = "../RAFT/raft-out/" + scene + "/" + str(fr_num-1) + ".npy"
    raft = np.load(RAFT_PATH)[2:-2, :, :]

    size = np.shape(gt)
    size = (size[1], size[0]*3)
    # print(size)

    for scale in {1.0, 0.25, 1.0/16}:
        tmp = np.concatenate((out, gt, raft), 0)
        print(np.shape(tmp))
        vis(tmp, scale)
        # vis(cv2.resize(zoom(tmp,  (scale, scale, 1)), size, interpolation=cv2.INTER_NEAREST), "{}.png".format(scale))
        # vis(cv2.resize(zoom(out,  (scale, scale, 1)), size, interpolation=cv2.INTER_NEAREST), "out-{}.png".format(scale))
        # vis(cv2.resize(zoom(gt,   (scale, scale, 1)), size, interpolation=cv2.INTER_NEAREST), "gt-{}.png".format(scale))
        # vis(cv2.resize(zoom(raft, (scale, scale, 1)), size, interpolation=cv2.INTER_NEAREST), "raft-{}.png".format(scale))

paint("shaman_2", 11)