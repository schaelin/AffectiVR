# 픽셀 값으로 눈 감을 때 탐지

import cv2
import csv
import numpy as np
np.seterr(invalid='ignore')
import os
from find_pupil import *

buffer = np.array([], dtype=np.float32)


def add_to_buffer(item):
    buffer_size = 30
    global buffer
    buffer = np.append(buffer, item)

    if len(buffer) > buffer_size:
        buffer = buffer[-buffer_size:]

video_path = "C:/Users/user/Desktop/newnew/P081/000/exports/000/eye1.mp4"
with tf.compat.v1.Session() as sess:
    # load best model
    logger = Logger("3A4Bh-Ref25", "INC", "", config, dir="models/")
    model = load_model(sess, "3A4Bh-Ref25", "INC", logger)

    # load the video or camera
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    ret = True

    while ret:
        ret, frame1 = cap.read()

        if ret:
            # Our operations on the frame come here
            frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame_copy = frame1.copy()
            frame_copy1 = frame_copy.copy()
            f_shape = frame.shape
            if frame.shape[0] != 192:
                frame = rescale(frame)

            image = gray_normalizer(frame)
            image = change_channel(image, config["input_channel"])
            [p] = model.predict(sess, [image])
            x, y, w = upscale_preds(p, f_shape)
            pupil_size = ((x, y), (w, w), 0)
            colors_red = (0, 0, 250)
            colors_white = (250, 250, 250)
            colors_green = (0, 250, 0)
            add_to_buffer([x, y])
            # print(buffer)
            colors = colors_green

            indices = np.where(frame_copy[:, :, 1] == 255)
            pixel_count = len(indices[0])
            brightness_values = frame_copy[indices][:, 0]
            mean_color = np.mean(brightness_values)
            # mean_color = np.mean(frame_copy1[indices], axis=0)
            # print(mean_color)
            add_to_buffer(mean_color)
            # print(pixel_count)
            # print(w)
            if np.isnan(buffer).any():
                colors = colors_red
                print('####', w)
            else:
                colors = colors_white


            cv2.ellipse(
                frame_copy,
                tuple(int(v) for v in pupil_size[0]),
                tuple(int(v / 2) for v in pupil_size[1]),
                pupil_size[2],
                0, 360,
                colors, -1
            )

            cv2.imshow("annotator", frame_copy)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()