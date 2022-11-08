import cv2
import numpy as np
import mediapipe as mp
import argparse
from os.path import join
from scipy.spatial import distance

# command line argument for crop level
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--crop', help="level of eye/iris crop 0-5 where 0 is iris", type=int, default=3)
parser.add_argument('-v', '--video', help="relative path of video source for analysis or 0 for webcam", default=0)
parser.add_argument('-s', '--save', help="if saving each frame cropped region is required", default=False)
args = parser.parse_args()

# mediapipe face mesh solution
mp_face_mesh = mp.solutions.face_mesh

# face mesh region indices for different eye zoom levels
INDICES = {
    # iris indices
    'LEYE_0': [474,475, 476, 477],
    'REYE_0': [469, 470, 471, 472],
    # eyes indices zoom 1
    'LEYE_1': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    'REYE_1': [133, 155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173],
    # eyes indices zoom 2
    'LEYE_2': [463, 341, 256, 252, 253, 254, 339, 255, 359, 467, 260, 259, 257, 258, 286, 414],
    'REYE_2': [243, 112, 26, 22, 23, 24, 110, 25, 130, 247, 30, 29, 27, 28, 56, 190],
    # eyes indices zoom 3
    'LEYE_3': [464, 453, 452, 451, 450, 449, 448, 261, 446, 342, 445, 444, 443, 442, 441, 413],
    'REYE_3': [244, 233, 232, 231, 230, 229, 228, 31, 226, 113, 225, 224, 223, 222, 221, 189],
    # eyes indices zoom 4
    'LEYE_4': [465, 357, 350, 349, 348, 347, 346, 340, 265, 353, 276, 283, 282, 295, 285, 417],
    'REYE_4': [245, 128, 121, 120, 119, 118, 117, 111, 35, 124, 46, 53, 52, 65, 55, 193],
    # eyes indices zoom 5
    'LEYE_5': [351, 412, 343, 277, 329, 330, 280, 352, 345, 372, 383, 300, 293, 334, 296, 336, 9, 8, 168],
    'REYE_5': [122, 188, 114, 47, 100, 101, 50, 123, 116, 143, 156, 70, 63, 105, 66, 107, 9, 8, 168]
}

# selected indices
LEYE = INDICES['LEYE_' + str(args.crop)]
REYE = INDICES['REYE_' + str(args.crop)]

# tracking each frame
frame_counter = 0

diam = []
# live webcam input
cap = cv2.VideoCapture(args.video)

# detect zoom level eyes
with mp_face_mesh.FaceMesh(refine_landmarks=bool(args.crop == 0)) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame config
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]

        # run mediapipe face mesh on each frame
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # array of every face mesh point coordinates detected on frame
            mesh_points=np.array([np.multiply([point.x, point.y], [frame_w, frame_h]).astype(int) for point in landmarks])
            frame_copy = frame.copy()
            mask = np.zeros(frame.shape, dtype=np.uint8)
            # draw eye regions on frame
            if args.crop == 0:
                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEYE])
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[REYE])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                cv2.circle(frame, center_left, int(l_radius), (0,255,0), 1, cv2.LINE_AA)
                cv2.circle(frame, center_right, int(r_radius), (0,255,0), 1, cv2.LINE_AA)
                if args.save:
                    cv2.circle(mask, center_right, int(r_radius), (255,255,255), -1)
            else:
                cv2.polylines(frame, [mesh_points[LEYE]], True, (0,255,0), 1, cv2.LINE_AA)
                cv2.polylines(frame, [mesh_points[REYE]], True, (0,255,0), 1, cv2.LINE_AA)
                if args.save:
                    cv2.drawContours(mask, [mesh_points[LEYE]], -1, (255, 255, 255), -1, cv2.LINE_AA)

            # crop out eye regions
            if mask.any():
                roi = cv2.bitwise_and(frame_copy, mask)
                gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                x,y,w,h = cv2.boundingRect(gray_mask)
                result = roi[y:y+h,x:x+w]
                new_mask = gray_mask[y:y+h,x:x+w]
                result[new_mask==0] = (0,0,0)

                # comment out this if block if don't want to perform pupil segmentation
                if args.crop==0:                    

                    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
                    equ = cv2.equalizeHist(gray)
                    # clahe = cv2.createCLAHE(clipLimit=2.0)
                    # cl1 = clahe.apply(gray)
                    gray = cv2.GaussianBlur(equ,(7,7),0)
                    gray = cv2.medianBlur(gray,7)
                    kernel = np.ones((7,7),np.uint8)
                    kernel2 = np.ones((12,12),np.uint8)
                    _,thresh = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY_INV)
                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
                    edges = cv2.dilate(cv2.Canny(closing,0,255),None)
                    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-3:]

                    image_center = np.asarray(edges.shape) / 2
                    image_center = tuple(image_center.astype('int32'))

                    cns = []

                    for contour in cnt:
                        # find center of each contour
                        M = cv2.moments(contour)
                        center_X = int(M["m10"] / M["m00"])
                        center_Y = int(M["m01"] / M["m00"])
                        contour_center = (center_X, center_Y)
                        
                        # calculate distance to image_center
                        distances_to_center = (distance.euclidean(image_center, contour_center))
                        
                        cns.append({'contour': contour, 'center': contour_center, 'distance_to_center': distances_to_center})

                    closest_cnts = sorted(cns, key=lambda i: i['distance_to_center'])
                    cnt= closest_cnts[0]['contour']


                    (x,y),radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x),int(y))
                    radius = int(radius)

                    x = 11.7 / (int(r_radius)*2)
                    diam.append(radius*2*x)
                    cv2.circle(result,center,radius,(0,255,0),1)


                # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

                if args.save:
                    # cv2.imwrite(join(args.save, str(frame_counter) + '.png'), edges)
                    cv2.imwrite(join(args.save, str(frame_counter) + '.png'), result)

            frame_counter +=1
            
        cv2.imshow('img', frame) 
        key = cv2.waitKey(1)
        if key ==ord('q'):
            break
cap.release()
np.savetxt(join(args.save[0], 'sizes.csv'), np.array(diam), delimiter=',')
cv2.destroyAllWindows()