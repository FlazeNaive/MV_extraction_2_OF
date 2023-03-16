import os
import numpy as np
import cv2 

import glob
import flowiz as fz
import matplotlib.pyplot as plt

#========== FUNCTIONS ==========

def get_flow_from_mv(mvs_0): 
    # motion vectors for blocks
    flow = np.swapaxes([mvs_0[:, 3], mvs_0[:, 4],   # start point (X0, Y0)
                        mvs_0[:,5] - mvs_0[:,3],    # delta X
                        mvs_0[:, 6] - mvs_0[:, 4]], # delta Y
                        0, 1)
    # print(np.shape(flow))
    return flow

def vis_flow(flow):
    vis_mat = np.zeros((h, w, 2))

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
                    vis_mat[cur[0]][cur[1]] = (mv[0][2], mv[0][3])
                except:
                    continue
    return vis_mat


# ================ draw vectors ================
def draw_vector(flow, draw_img, OUTPUTFILE):
    for mv in np.split(flow, num_vec):
        # print("---")
        start_pt = (mv[0][0], mv[0][1])
        end_pt = (mv[0][2] + start_pt[0], mv[0][3] + start_pt[1]) 
        cv2.arrowedLine(draw_img, start_pt, end_pt, (255, 0, 0), 1, cv2.LINE_AA, 0, 0.1)
    
    cv2.imwrite(OUTPUTFILE, draw_img)

# ================ save *.flo ================
def save_to_flo(vis_mat, filename): 
    # mkdir if not exist
    filepath = os.path.dirname(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
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

# ================ save visualized flow ================

def save_flo_to_visual(input_filename, output_filename):
    # mkdir if not exist
    filepath = os.path.dirname(output_filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    files = glob.glob(input_filename)
    img = fz.convert_from_file(files[0])
    plt.imshow(img)

    plt.savefig(output_filename)


# ========== MAIN ==========

# FILEPATH = "data-extra/alley_1/out-2023-03-10T14:52:02/motion_vectors/"
FRAMEPATH = "data-extra/alley_1/out-2023-03-10T14:52:02/frames/"

# get height and width of frame
frame_0 = cv2.imread(FRAMEPATH + "frame-0.jpg")
print(np.shape(frame_0))
(h, w, _) = np.shape(frame_0)

INPUT_BASE = "data-extra/"
OUTPUT_BASE = "data-output/"
GT_BASE = "data-gtflow/"

for FILE in os.listdir(INPUT_BASE):
    PATH1 = INPUT_BASE + FILE + "/"
    for random_name in os.listdir(PATH1):
        PATH2 = PATH1 + random_name + "/"
        print(PATH2)

        MVPATH = PATH2 + "motion_vectors/"
        TYPES = PATH2 + "frame_types.txt"

        with open(TYPES, "r") as f:
            # read lines until the end of the file
            frame_counter = 0
            while True:
                frame_type = f.readline()

                if not frame_type:
                    break
                frame_type = frame_type.strip()

                #process the P frame, convert to flow
                if frame_type == "P":
                    # read the motion vectors
                    mv_file = MVPATH + "mvs-" + str(frame_counter) + ".npy"
                    mvs = np.load(mv_file)

                    #get flow from motion vectors
                    flow = get_flow_from_mv(mvs)

                    # draw the flow matrix
                    vis_mat = vis_flow(flow)

                    #save the flow matrix
                    save_to_flo(vis_mat, OUTPUT_BASE + "mv-flo/" + 
                                    FILE + "/flow-" + str(frame_counter) + ".flo")

                    # convert int to 4-number string using formatted string
                    print("Current frame #: ", frame_counter)
                    frame_counter_str = f"{frame_counter:04d}"
                    print(frame_counter_str)

                    #read ground truth flow
                    gt_flow = read_flo_file(GT_BASE + FILE + "/frame_" + frame_counter_str + ".flo")

                    #combine ground truth flow and predicted flow
                    combined_flow = np.concatenate((gt_flow, vis_mat), axis=1)

                    #save the combined flow
                    save_to_flo(combined_flow, OUTPUT_BASE + "combined-flo/" + 
                                FILE + "/combined-" + str(frame_counter) + ".flo")
                    
                    #save the visualized flow
                    save_flo_to_visual(OUTPUT_BASE + "combined-flo/" + 
                                FILE + "/combined-" + str(frame_counter) + ".flo",
                                OUTPUT_BASE + "vis-flo/" + 
                                FILE + "/vis-" + str(frame_counter) + ".png"
                                )

                frame_counter = frame_counter + 1
        quit()
                    


quit()


# print("draw")
# print(np.shape(draw_img))

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
