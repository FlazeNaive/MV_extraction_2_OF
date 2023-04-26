import numpy as np
import cv2
import torch

ID = '1'

def get_flow_from_mv(mvs_0): 
    # motion vectors for blocks
    flow = np.swapaxes([mvs_0[:, 3], mvs_0[:, 4],   # start point (X0, Y0)
                        mvs_0[:,5] - mvs_0[:,3],    # delta X
                        mvs_0[:, 6] - mvs_0[:, 4]], # delta Y
                        0, 1)
    # print(np.shape(flow))
    return flow

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


def get_mask_2(shape, mvs):
    mask = np.zeros(shape, dtype=np.float32)
    print("shape: ", shape)
    for mv in mvs:
        start_pt = (mv[3], mv[4])
        end_pt = (mv[5], mv[6])
        for i in range(16):
            for j in range(16):
                try:
                    # cur = (end_pt[1] + i - 8, end_pt[0] + j - 8) # pixel currently on
                    cur = (start_pt[1] + i - 8, start_pt[0] + j - 8) # pixel currently on
                    mask[cur[0]][cur[1]] += 1
                except:
                    # print("(i, j) = ", cur)
                    # wait = input()
                    continue       

    return mask


def vis_flow(h, w, flow):
    vis_mat = np.zeros((h, w, 2), dtype=np.uint8)
    # cv2.imshow("flow_viw",vis_mat[:,:,0]*10)
    # cv2.waitKey(0)


    num_vec = np.shape(flow)[0]
    for mv in np.split(flow, num_vec):
        # print("---")
        start_pt = (mv[0][0], mv[0][1])
        end_pt = (mv[0][2] + start_pt[0], mv[0][3] + start_pt[1]) 

        #draw motion vector blocks
        for i in range(16):
            for j in range(16):
                try:
                    cur = (end_pt[1] + i - 8, end_pt[0] + j - 8) # pixel currently on
                    vis_mat[cur[0]][cur[1]] = (255, 255) #(mv[0][2], mv[0][3])
                except:
                    continue
    return vis_mat

# ================ draw vectors ================
def draw_vector(mvs, draw_img):
    for mv in mvs:
        # print("---")
        start_pt = (mv[3], mv[4])
        end_pt = (mv[5], mv[6])
        cv2.arrowedLine(draw_img, start_pt, end_pt, (255, 0, 0), 1, cv2.LINE_AA, 0, 0.1)
        cv2.circle(draw_img, end_pt, 1, (0, 0, 255), 1, cv2.LINE_AA, 0)
    
    # cv2.imshow("draw_img", draw_img)
    # cv2.waitKey(0)

img = cv2.imread("frame-" + ID + ".jpg")
print("imgshape: ",img.shape)
# cv2.imshow("img", img)
# cv2.waitKey(0)


mask_shape = np.shape(img)[0:2]

print(mask_shape)

mvs = np.load("mvs-"+ID+".npy")
flow = get_flow_from_mv(mvs)
mask = get_mask_2(mask_shape, mvs)
print(np.shape(mask))
scaled = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
cv2.imwrite("mask.png", scaled * 255)
# cv2.imshow("mask", mask)
# cv2.waitKey(0)

# flow_vis = vis_flow(mask_shape[0], mask_shape[1], flow)
# cv2.imshow("flow_viw",flow_vis[:,:,0]*20)
# cv2.waitKey(0)

# canvas = np.zeros(mask_shape, dtype=np.uint8)
# canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
# fusion = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
# cv2.imshow("masked", fusion) 
# cv2.waitKey(0)
# draw_vector(mvs, canvas)

def calc_loss(input, mask):
    loss = np.sqrt(np.sum((input- mask)**2))
    return loss
    

gt = read_flo_file("frame_0006.flo")
res = read_flo_file("flow-6.flo")
raft = np.load("raft.npy")[2:438, :, :]

print("gt shape: ", gt.shape)
print("res shape: ", res.shape)
print("raft shape: ", raft.shape)

gt = np.multiply(gt, np.stack((mask, mask), axis=2))
res = np.multiply(res, np.stack((mask, mask), axis=2))
raft = np.multiply(raft, np.stack((mask, mask), axis=2))

count = np.sum(mask)
print("count: ", count)


loss = calc_loss(gt, res)
print("MV:\n\tsum = ", loss, "\n\tmean = ", loss / count)
loss = calc_loss(raft, res)
print("RAFT:\n\tsum = ", loss, "\n\tmean = ", loss / count)

# epe = torch.sum((res - gt)**2, dim=0).sqrt()


# yufan@128.179.183.30