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
    img = fz.convert_from_flow(a)
    OUTPATH = "tmp_visual/"
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    cv2.imwrite(OUTPATH + str(name) + ".png", img)

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

    for scale in {1.0, 0.25, 1.0/16}:
        vis(zoom(out, (scale, scale, 1)), "out-{}.png".format(scale))
        vis(zoom(gt, (scale, scale, 1)), "gt-{}.png".format(scale))
        vis(zoom(raft, (scale, scale, 1)), "raft-{}.png".format(scale))

paint("bamboo_1", 5)