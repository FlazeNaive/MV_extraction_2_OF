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
    mask = np.zeros((h, w))

    num_vec = np.shape(flow)[0]
    for mv in np.split(flow, num_vec):
        # print("---")
        start_pt = (mv[0][0], mv[0][1])
        end_pt = (mv[0][2] + start_pt[0], mv[0][3] + start_pt[1]) 

        #draw motion vector blocks
        for i in range(16):
            for j in range(16):
                try:
                    # cur = (end_pt[1] + i - 8, end_pt[0] + j - 8) # pixel currently on
                    cur = (start_pt[1] + i - 8, start_pt[0] + j - 8) # pixel currently on
                    vis_mat[cur[0]][cur[1]] = (mv[0][2], mv[0][3])
                    mask[cur[0]][cur[1]] = 255
                except:
                    continue
    return vis_mat, mask


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

def save_flo_to_visual(input_filename, frame, output_filename, mask = None, method = "no_mask"):
    # mkdir if not exist
    filepath = os.path.dirname(output_filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    files = glob.glob(input_filename)
    img = fz.convert_from_file(files[0])
    
    if method == "no_mask": 
        plt.subplot(1, 2, 1)
        plt.imshow(img)

        plt.subplot(1, 2, 2)
    else: 
        plt.subplot(1, 3, 1)
        plt.imshow(img)

        plt.subplot(1, 3, 2)
        plt.imshow(mask)

        plt.subplot(1, 3, 3)
    # convert img read by OpenCV to RGB
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # change to tight layout
    plt.tight_layout()
    # save high resolution image 
    plt.savefig(output_filename, dpi=300)


# ========== MAIN ==========

INPUT_BASE = "data-extra/"
OUTPUT_BASE = "data-output/"
GT_BASE = "data-gtflow/"

for FILE in os.listdir(INPUT_BASE):
    # if (FILE != "alley_1"):
    #     continue
    PATH1 = INPUT_BASE + FILE + "/"

    for random_name in os.listdir(PATH1):
        PATH2 = PATH1 + random_name + "/"
        print(PATH2)

        FRAMEPATH = PATH2 + "/frames/"
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
                    # read the frame
                    frame_file = FRAMEPATH + "frame-" + str(frame_counter) + ".jpg"
                    # get height and width of frame
                    frame_img = cv2.imread(frame_file)
                    (h, w, _) = np.shape(frame_img)

                    # read the motion vectors
                    mv_file = MVPATH + "mvs-" + str(frame_counter) + ".npy"
                    mvs = np.load(mv_file)

                    #get flow from motion vectors
                    flow = get_flow_from_mv(mvs)

                    # draw the flow matrix
                    vis_mat, mask_mat = vis_flow(flow)

                    #save the flow matrix
                    save_to_flo(vis_mat, OUTPUT_BASE + FILE + "/mv-flo/" + 
                                    "flow-" + str(frame_counter) + ".flo")

                    # convert int to 4-number string using formatted string
                    print("Current frame #: ", frame_counter)
                    frame_counter_str = f"{frame_counter:04d}"
                    print(frame_counter_str)

                    #read ground truth flow
                    gt_flow = read_flo_file(GT_BASE + FILE + "/frame_" + frame_counter_str + ".flo")

                    #combine ground truth flow and predicted flow
                    combined_flow = np.concatenate((gt_flow, vis_mat), axis=0)

                    #save the combined flow
                    save_to_flo(combined_flow, OUTPUT_BASE + FILE + "/combined-flo" + 
                                "/combined-" + str(frame_counter) + ".flo")
                    
                    #save the visualized flow
                    save_flo_to_visual(OUTPUT_BASE + FILE + "/combined-flo/" + 
                                                "/combined-" + str(frame_counter) + ".flo",
                                        frame_img, 
                                        OUTPUT_BASE + FILE + "/vis-flo/" + 
                                                "/vis-" + str(frame_counter) + ".png",
                                        mask_mat,
                                        "no_mask"
                                )
                    
                    #save the mask
                    PATH_MASK = OUTPUT_BASE + FILE + "/mask/"
                    if not os.path.exists(PATH_MASK):
                        os.makedirs(PATH_MASK)
                    cv2.imwrite(PATH_MASK + 
                                    "mask-" + str(frame_counter) + ".png"
                                    , mask_mat)

                frame_counter = frame_counter + 1

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
