import cv2
import numpy as np
import os
from scipy import interpolate
from find_pupil import *
from utils import change_channel, gray_normalizer


def add_to_buffer(item):
    buffer_size = 5
    global buffer
    buffer = np.append(buffer, item)

    if len(buffer) > buffer_size:
        buffer = buffer[-buffer_size:]



def pupil_detect(frame, sess, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f_shape = gray.shape
    frame_copy = frame.copy()

    if gray.shape[0] != 192:
        gray = rescale(gray)

    image = gray_normalizer(gray)
    image = change_channel(image, config["input_channel"])


    [p] = model.predict(sess, [image])
    x, y, w = upscale_preds(p, f_shape)

    pupil_size = ((x, y), (w, w), 0)

    indices = np.where(frame_copy[:, :, 1] == 255)
    brightness_values = frame_copy[indices][:, 0]
    mean_color = np.mean(brightness_values)
    add_to_buffer(mean_color)

    if np.isnan(buffer).any():
        pupil_size = ((0, 0), (0, 0), 0)
    else:
        pupil_size = ((x, y), (w, w), 0)

    return pupil_size


def eye_open(frame, sess, model, iriscode = False):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    is_erased = 0
    ellipse = pupil_detect(frame, sess, model)
    
    if ellipse[1][0] < 33:
        blink = 1
    else:
        blink = 0


    if blink == 0:
        gray, is_erased = remove_light(gray, ellipse)

    if iriscode == True:
        iriscode = make_iriscode(gray, ellipse)
        return blink, gray, iriscode, is_erased
    
    return blink, gray, is_erased


#조명 검출
def light_detect(gray):
    parameter = cv2.SimpleBlobDetector_Params()
    parameter.filterByColor = 255
    parameter.blobColor = 255
    parameter.minThreshold = 190
    parameter.minArea = 1  # this value defines the minimum size of the blob
    parameter.maxArea = 50  # this value defines the maximum size of the blob
    parameter.minConvexity = 0.0001
    parameter.minCircularity = 0.001
    
    light_frame = gray.copy()
    detector_ = cv2.SimpleBlobDetector_create(parameter)
    keypoints = detector_.detect(gray)
    points = cv2.KeyPoint_convert(keypoints)

    if len(points) != 0:
        for cx,cy in points:
            cv2.circle(light_frame,(int(cx),int(cy)), 6, color= (255),thickness=-1)

    return light_frame

#홍채 검출
def iris_detection(ellipse):
    iris_frame = np.zeros((480,640), dtype = "uint8")

    cv2.ellipse(
    iris_frame,
    tuple(int(v) for v in ellipse[0]),
    # tuple(int(v / 2 + 55) for v in ellipse["axes"]),
    (80,80),
    ellipse[2],
    0, 360, # start/end angle for drawing
    255, thickness=-1 # color (BGR): red
    )

    return iris_frame

def remove_light(gray, ellipse):
    light_frame = light_detect(gray)
    iris_frame  = iris_detection(ellipse)

    dst1 = cv2.inRange(light_frame, (140),(255))
    dst2 = cv2.inRange(iris_frame, 255, 255)
    mask = cv2.bitwise_and(dst1, dst1, mask=dst2)

    index_mask = np.where(mask == 255)
    is_erased = False

    if len(index_mask) != 0:
        is_erased = True
        for i in range(len(index_mask[1])):
            h = index_mask[0][i]
            w = index_mask[1][i]

            pixel_near_mask = []
            pixel_near_ = []
            pixel_near_index = []
            pixel_near = []

            pixel_near_mask = mask[h-3 : h + 4, w - 3 : w + 4]
            pixel_near_ = gray[h-3 : h + 4, w - 3 : w + 4]
            
            pixel_near_index = np.where(pixel_near_mask != 255)
            pixel_near = pixel_near_[pixel_near_index]


            # pixel_near_ = pixel_near[np.where(mask < 200)] 
            if len(pixel_near) < 1:
                gray[h][w] = np.median(pixel_near_)
            else:
                gray[h][w] = np.median(pixel_near)
    else:
        is_erased = False

    return gray, is_erased

#홍채 코드 그리기
def make_iriscode(gray, ellipse):

    inter_result = np.zeros((40,360), dtype = "uint8")

    iris_frame  = iris_detection(ellipse)

    cv2.ellipse(
    iris_frame,
    tuple(int(v) for v in ellipse[0]),
    tuple(int(v / 2 ) for v in ellipse[1]),
    ellipse[2],
    0, 360, # start/end angle for drawing
    0, thickness= -1 # color
    )

    mask = cv2.bitwise_and(gray, iris_frame, mask=iris_frame)
    mask[mask==0] = 255

    center = tuple(int(v) for v in ellipse[0])     # Detected center of the iris
    inner_radius = tuple(int(v / 2 ) for v in ellipse[1])      # Detected inner boundary radius
    outer_radius = 80     # Detected outer boundary radius

    if center[0] > 100 and center[1] > 100: 
        roi_iris = mask[center[1] - 100 : center[1] +  101, center[0] - 100 : center[0] +  101]
        roi_iris[100][100] = 255

        polar_image = convert_to_polar_image(roi_iris, (100,100), min(inner_radius[0], inner_radius[1]), outer_radius)
        index_mask = np.where(polar_image == 0)

        if len(polar_image) != 0:
            for i in range(len(index_mask[1])):
                h = index_mask[0][i]
                w = index_mask[1][i]

                pixel_near_mask = []
                pixel_near_ = []
                pixel_near_index = []
                pixel_near = []

                pixel_near_ = polar_image[h-1 : h + 2, w - 1 : w + 2]
              
                pixel_near_index = np.where(pixel_near_ != 0)

                if len(pixel_near_) == 0:
                    polar_image[h][w] = 255
                else:
                    pixel_near = pixel_near_[pixel_near_index]
                    polar_image[h][w] = np.median(pixel_near)
        
        # dst1 = cv2.inRange(polar_image, (130),(255))

        # k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # dst1 = cv2.dilate(dst1, k)

        # eyelid_mask = np.where(dst1 == 255)
        # polar_image[eyelid_mask] = 255

        # cv2.imshow("fgfgfg",polar_image)

        
        for i in range(360):
            line_data = polar_image[:,i]
            line_data_1 = line_data[np.where(line_data != 255)]

            x_len= len(line_data_1)
            x = np.linspace(0, 39, x_len)
            
            fq = interpolate.interp1d(x,line_data_1,kind = 'linear')
            xint = np.linspace(x.min(), x.max(), 40)
            
            new_line_data = fq(xint)
            inter_result[:,i] = new_line_data

        result = cv2.equalizeHist(inter_result)
        return inter_result
    

def cartesian_to_polar(x, y, center_x, center_y):
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    theta = np.arctan2(y - center_y, x - center_x)
    return r, theta


def convert_to_polar_image(image, center, inner_radius, outer_radius):
    cx, cy = center
    height = outer_radius - inner_radius
    polar_image = np.zeros((height, 360), dtype=np.uint8)
    i = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            dist, angle = cartesian_to_polar(j, i, cx, cy)
            if inner_radius <= dist < outer_radius:
                polar_image[int(dist - inner_radius), int(np.degrees(angle))] = image[i, j]
    return polar_image


if __name__ == '__main__':
    sess = tf.compat.v1.Session()
    model = load_model(sess, "3A4Bh-Ref25", "INC")
    buffer = np.array([], dtype=np.float32)

    blink = 0
    num = 0

    ##########################################################
    path = './001.mp4'
    cap = cv2.VideoCapture(path)
    # cap = cv2.VideoCapture(0)

    if cap.isOpened() == False:
        print("Unable to read camera")

    while True:
        # get every frame from the web-cam
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, dsize=(640,480), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("gray", gray)
        iriscode = None
        
        ########if you want to see the iriscode, you can use this code
        # blink, gray, iriscode, is_erased = eye_open(frame, sess, model, iriscode = True)

        # if blink == 0:
        #     cv2.imshow("iris", iriscode)
        #     cv2.imshow("gray_done", gray)

        #########if you don't want to see the iriscode, you can use this code
        blink, gray, is_erased = eye_open(frame, sess, model)

        if blink == 0:
            cv2.imshow("gray_done", gray)

        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


    cv2.destroyAllWindows()




