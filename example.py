#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()


orig = cv2.imread("ivan.jpg")
image0 = cv2.imread("ivan.jpg")
image1 = cv2.imread("ivan.jpg")

crop_img = orig[90:90+70,270:270+60]
image1[90:90+70, 270:270+60,:] = 0

# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
for curr_image in [image0, image1]:
    result = detector.detect_faces(curr_image)

    if len(result) > 0:
        bounding_box = result[0]['box']
        keypoints = result[0]['keypoints']
        found_color = (255,150,75)

        cv2.rectangle(curr_image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                      found_color,
                      2)


        cv2.circle(curr_image,(keypoints['left_eye']), 2, found_color, 2)
        cv2.circle(curr_image,(keypoints['right_eye']), 2, found_color, 2)
        cv2.circle(curr_image,(keypoints['nose']), 2, found_color, 2)
        cv2.circle(curr_image,(keypoints['mouth_left']), 2, found_color, 2)
        cv2.circle(curr_image,(keypoints['mouth_right']), 2, found_color, 2)
        confStr = "{:.2f}%".format(result[0]["confidence"]*100)
        cv2.putText(curr_image, confStr, (bounding_box[0]-3, bounding_box[1]-5),1, 1.05,found_color,2)
    else:
        cv2.rectangle(curr_image,
                  (270, 90),
                  (270 + 60, 90 + 70),
                  (0, 155, 255),
                  2)
        confStr = "{:.2f}%".format(0)
        cv2.putText(curr_image, confStr, (270-3, 90-5),1, 1.05,(0,175,255),2)
    print(result)

cv2.imshow("ivan_found", image0)
cv2.imshow("ivan_mask", image1)
cv2.imshow("orig",orig)
cv2.imshow("crop",crop_img)
cv2.waitKey(0)


