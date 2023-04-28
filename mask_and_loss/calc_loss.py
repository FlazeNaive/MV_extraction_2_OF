import numpy as np
import cv2
import os
import json
from scipy.ndimage import zoom
from skimage.transform import resize, rescale

import flowiz

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


def vis(a, b, name="default"):
    c = np.concatenate((a, b), 0)
    img = flowiz.convert_from_flow(c)
    OUTPATH = "tmp_visual/"
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    cv2.imwrite(OUTPATH + str(name) + ".png", img)
    

def calc_loss(out, gt, mask, scale = 1.0, name="default"):
    # import ipdb; ipdb.set_trace()
    out = zoom(out, (scale, scale, 1))
    gt = zoom(gt, (scale, scale, 1))
    mask = zoom(mask, scale)

    # h, w, _ = out.shape
    # print(h, w, _)
    # out = rescale(out, scale)
    # gt = rescale(gt, scale)
    # mask= rescale(mask, scale)

    # # out = resize(out, (h * scale, w * scale, 2))
    # # gt = resize(gt, (h * scale, w * scale, 2))
    # # mask = resize(mask, (h*scale, w*scale))
    # print(out.shape)
    # print(mask)
    mask = np.where(mask > 127, 1, 0)

    # vis(out, gt, name+str(scale))
    
    gt = np.multiply(gt, np.stack((mask, mask), axis=2))
    out= np.multiply(out, np.stack((mask, mask), axis=2))
    # import ipdb; ipdb.set_trace()
    sum_up = np.sqrt(np.sum((out - gt)**2, axis=-1)).sum()
    cnt_pix = np.sum(mask)
    print(sum_up, cnt_pix, sum_up / cnt_pix)
    return (sum_up/cnt_pix)


def calc_frame(scene, fr_num):
    MASK_PATH = "../data-output/" + scene + "/mask/mask-" + str(fr_num) + ".png"
    mask = cv2.cvtColor(cv2.imread(MASK_PATH), cv2.COLOR_BGR2GRAY)
    # print("Loading OUTPUT...")
    OUT_PATH = "../data-output/" + scene + "/mv-flo/flow-" + str(fr_num) + ".flo"
    out = read_flo_file(OUT_PATH)
    # print("Loading GT...")
    GT_PATH = "../data-gtflow/" + scene + "/frame_" + str(fr_num).zfill(4) + ".flo"
    gt = read_flo_file(GT_PATH)
    # print("Loading RAFT...")
    RAFT_PATH = "../../RAFT/raft-out/" + scene + "/" + str(fr_num-1) + ".npy"
    raft = np.load(RAFT_PATH)[2:-2, :, :]

    out_loss = []
    raft_loss = []

    # out_loss = [2, 3, 3]
    # raft_loss = [2, 3, 3]

    for scale in (1.0, 0.25, 1.0/16):
        out_loss.append(calc_loss(out, gt, mask, scale, scene+"-"+str(fr_num)+"-out"))
        raft_loss.append(calc_loss(raft, gt, mask, scale, scene+"-"+str(fr_num)+"-raft"))

    return (out_loss, raft_loss)

# print(calc_frame("alley_1", 2))
print(calc_frame("bamboo_1", 5))
quit()

def calc_scene(scene):
    PATH1 = "../data-extra/" + scene + "/"
    print("processing: ", scene)
    rec = []

    for random_name in os.listdir(PATH1):
        PATH2 = PATH1 + random_name + "/"
        TYPES = PATH2 + "frame_types.txt"

        frame_counter = 0
        P_frame_counter = 0
        sum_loss_out = np.array((0, 0, 0), dtype = 'float64')
        sum_loss_raft= np.array((0, 0, 0), dtype = 'float64')

        with open(TYPES, "r") as f:
            # read lines until the end of the file
            while True:
                frame_type = f.readline()

                if not frame_type:
                    break
                frame_type = frame_type.strip()

                #process the P frame, convert to flow
                if frame_type == "P":
                    print("frame # = ", frame_counter)
                    (a, b) = calc_frame(scene, frame_counter)
                    # print(a)
                    # print(b)
                    rec.append((a, b))
                    sum_loss_out += a
                    sum_loss_raft += b
                    P_frame_counter += 1
                
                frame_counter += 1
        
        ave_loss = ((sum_loss_out/P_frame_counter).tolist(), (sum_loss_raft/P_frame_counter).tolist())
        print("average: ", ave_loss)
        np.save("../loss/" + scene + ".npy", rec)
        return ave_loss

# print(calc_scene("alley_1"))
if __name__ == '__main__':
    DS_PATH = "../data-output/"
    dict = []
    for scene in os.listdir(DS_PATH):
        scene_res = calc_scene(scene)
        cur = {scene: {"MV": scene_res[0], "RAFT": scene_res[1]}}
        print(json.dumps(cur))
        dict.append(cur)

    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)