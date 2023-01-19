import colorsys
import cv2
import numpy as np
import os

MIN_MATCH_COUNT = 10
sift = cv2.SIFT_create(contrastThreshold=0.09, edgeThreshold=10)

# Precalculate keypoints and descriptors for all template object images
template_root_path = 'project2/image/template'
objects = dict() # {'name': list((keypoint, descriptor))}

for name in os.listdir(template_root_path):
    objects[name] = list()

    for img_file in os.listdir(f'{template_root_path}/{name}'):
        img_gray = cv2.imread(f'{template_root_path}/{name}/{img_file}', cv2.IMREAD_GRAYSCALE)

        kp, des = sift.detectAndCompute(img_gray, None)
        objects[name].append((kp, des))

print(objects.keys())

# Get n distinct colors for each object bounding box
def hsv2rgb(h, s, v): 
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
    return (int(255*r), int(255*g), int(255*b)) 

huePartition = 1.0 / (len(objects) + 1) 
object_box_colors = [hsv2rgb(huePartition * value, 1.0, 1.0) for value in range(len(objects))]
object_box_colors = dict(zip(objects.keys(), object_box_colors))

# Function get matches between 2 list of descriptors
def get_matches(query_des, template_des):
    # Initialize and match keypoint descriptors using Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    # matches = bf.knnMatch(query_des, template_des, k=2)
    matches = bf.knnMatch(template_des, query_des, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)
    
    return good_matches

# Read query image
query_file_path = 'project2/image/query/scene1.jpg'
query_img = cv2.imread(query_file_path)
query_gray_img = cv2.imread(query_file_path, cv2.IMREAD_GRAYSCALE)
print('\n%s' % query_file_path)

# Calculate keypoints and descriptors for query image
query_kp, query_des = sift.detectAndCompute(query_gray_img, None)

# Check matching between query and each object in objects
for object_key in objects.keys():
    best_idx = -1
    best_matches = []

    for idx, (template_kp, template_des) in enumerate(objects[object_key]):
        matches = get_matches(query_des, template_des)
        if len(matches) > len(best_matches):
            best_idx = idx
            best_matches = matches
    
    if len(best_matches) >= MIN_MATCH_COUNT:
        print('%-16s %d keypoints match' % (object_key, len(best_matches)))
        template_kp, template_des = objects[object_key][best_idx]

        # Find the homography matrix
        template_pts = np.float32([template_kp[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        query_pts = np.float32([query_kp[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(template_pts, query_pts, cv2.RANSAC, 5.0)

        if M is None:
            print('%-16s Not enought inliers to calculate homography matrix' % '')
            continue

        # Transform object keypoints to scene coordinates
        template_pts = np.float32([template_kp[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        template_pts = cv2.perspectiveTransform(template_pts, M)

        # Find the bounding box coordinates
        x, y, w, h = cv2.boundingRect(template_pts)

        # Draw the bounding box on the scene image
        cv2.rectangle(query_img, (x, y), (x+w, y+h), object_box_colors[object_key], 6)
        cv2.putText(query_img, object_key, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 3, object_box_colors[object_key], 6)
    else:
        print('%-16s Not enough keypoints match are found - %d/%d' % (object_key, len(best_matches), MIN_MATCH_COUNT))

scale = 1/4
query_img = cv2.resize(query_img, None, fx=scale, fy=scale)
cv2.imshow("SIFT Detector", query_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
