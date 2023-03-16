import os
import numpy as np
import cv2 

import glob
import flowiz as fz
import matplotlib.pyplot as plt

# PATH = "dataset/alley_1.MPEG"

FILEID = "1"

FILEPATH = "data-extra/alley_1/out-2023-03-10T14:52:02/motion_vectors/"
FRAMEPATH = "data-extra/alley_1/out-2023-03-10T14:52:02/frames/"

frame_0 = cv2.imread(FRAMEPATH + "frame-" + FILEID + ".jpg")
print(np.shape(frame_0))
(h, w, _) = np.shape(frame_0)

# draw_img = np.zeros((w, h))
draw_img = np.zeros(np.shape(frame_0))

print("draw")
print(np.shape(draw_img))

mvs_0 = np.load(FILEPATH + "/mvs-" + FILEID + ".npy")
# print(mvs_0)
print(mvs_0[0])
print(mvs_0[-1])

# motion vectors for blocks
flow = np.swapaxes([mvs_0[:, 3], mvs_0[:, 4],   # start point (X0, Y0)
                    mvs_0[:,5] - mvs_0[:,3],    # delta X
                    mvs_0[:, 6] - mvs_0[:, 4]], # delta Y
                    0, 1)
# flow = np.ndarray([mvs_0[:,5] - mvs_0[:,3], mvs_0[:, 6] - mvs_0[:, 4]])
print(np.shape(flow))

vis_mat = np.zeros((h, w, 2))

num_vec = np.shape(flow)[0]
for mv in np.split(flow, num_vec):
    # print("---")
    start_pt = (mv[0][0], mv[0][1])
    end_pt = (mv[0][2] + start_pt[0], mv[0][3] + start_pt[1]) 
    # if start_pt[1]-8 > 436 or start_pt[0]-8> 1024:
    #     print("START:")
    #     print(start_pt)
    # if end_pt[1]-8 > 436 or end_pt[0]-8 > 1024:
    #     print("END:")
    #     print(end_pt)

    #draw motion vector blocks
    for i in range(16):
        for j in range(16):
            try:
                cur = (end_pt[1] + i - 8, end_pt[0] + j - 8) # pixel currently on
                # cur = (start_pt[1] + i - 24, start_pt[0] + j - 16)
                # draw_img[start_pt[1] + i - 16][start_pt[0] + j - 16] = (mv[0][2], mv[0][2], mv[0][3])
                # draw_img[end_pt[1] + i - 24][end_pt[0] + j - 8] = (mv[0][2], mv[0][2], mv[0][3])
                draw_img[cur[0]][cur[1]] = (mv[0][2], mv[0][2], mv[0][3])
                vis_mat[cur[0]][cur[1]] = (mv[0][2], mv[0][3])
                # draw_img[cur[0]][cur[1]] *= 20
                # draw_img[end_pt[1] + i - 16][end_pt[0] + j - 16] = (mv[0][2], mv[0][2], mv[0][3])
            except:
                # print("OUT OF RANGE")
                # occurs because the size is not dividable by 16
                # print(cur)
                continue
                # norm_image = cv2.normalize(draw_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # cv2.imwrite("flow.jpg", norm_image)
                # k = input()

# draw_img= cv2.normalize(draw_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16SC1)


# ================ draw vectors ================
# for mv in np.split(flow, num_vec):
#     # print("---")
#     start_pt = (mv[0][0], mv[0][1])
#     end_pt = (mv[0][2] + start_pt[0], mv[0][3] + start_pt[1]) 
#     cv2.arrowedLine(draw_img, start_pt, end_pt, (255, 0, 0), 1, cv2.LINE_AA, 0, 0.1)


cv2.imwrite("flow.bmp", draw_img)

# ================ save *.flo ================
def save_to_flo(vis_mat, filename): 
    height, width, nBands = vis_mat.shape
    assert nBands == 2, "Number of bands = %r != 2" % nBands
    u = vis_mat[: , : , 0]
    v = vis_mat[: , : , 1]	
    assert u.shape == v.shape, "Invalid flow shape"
    height, width = u.shape
    TAG_STRING = b'PIEH'

    f = open(filename,'wb')
    f.write(TAG_STRING)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)

    f.close()

# filename = "alley_" + FILEID + ".flo"
save_to_flo(vis_mat, "alley_" + FILEID + ".flo")

# ================ read *.flo ================
def read_flo_file(filename, memcached=False):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    if memcached:
        filename = io.BytesIO(filename)
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

gt_flow = read_flo_file('frame_0001.flo')

# ================ combine ================
showww = np.concatenate((vis_mat , gt_flow), axis=0)
save_to_flo(showww, "RUAURUA.flo")

# ================ flowiz ================
# img = fz.convert_from

filename = "RUAURUA.flo"
# files = glob.glob('demo/flo/*.flo')
files = glob.glob(filename)
img = fz.convert_from_file(files[0])
plt.imshow(img)

gt = cv2.imread('frame_0001.png')
plt.savefig('flow---.png')
