import cv2
import numpy as np
import os

class Model:
    def __init__(self):
        # self.MIN_MATCH_COUNT = 10
        self.min_match_thresh = {'cat': 8, 'calculator': 50}
        self.LOWE_RATIO = 0.6

        self.sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=30)

        self.labels = list()
        self.templates_features = dict() # {'label': list of (keypoints, descriptors) for each image in label dir}

        self.print_message = False

    def load_templates(self, template_root_path):
        for label in os.listdir(template_root_path):
            self.labels.append(label)
            self.templates_features[label] = list()

            for img_file in os.listdir(os.path.join(template_root_path, label)):
                img_gray = cv2.imread(os.path.join(template_root_path, label, img_file), cv2.IMREAD_GRAYSCALE)

                kp, des = self.sift.detectAndCompute(img_gray, None)
                self.templates_features[label].append((kp, des))

        if self.print_message: print(self.labels)

    def get_matches(self, query_des, template_des):
        # Brute Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = bf.knnMatch(template_des, query_des, k=2)

        # FLANN Matcher
        # FLANNINDEXKDTREE = 1
        # index_params = dict(algorithm=FLANNINDEXKDTREE, trees=5)
        # search_params = dict(checks=50)
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(template_des, query_des, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.LOWE_RATIO * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def predict(self, query_gray_img):
        query_kp, query_des = self.sift.detectAndCompute(query_gray_img, None)
        prediction = dict.fromkeys(self.labels)

        for label in self.labels:
            max_idx = -1
            max_matches = []

            for idx, (template_kp, template_des) in enumerate(self.templates_features[label]):
                matches = self.get_matches(query_des, template_des)
                if len(matches) > len(max_matches):
                    max_idx = idx
                    max_matches = matches

            # if len(max_matches) >= self.MIN_MATCH_COUNT:
            if len(max_matches) >= self.min_match_thresh[label]:
                if self.print_message: print('%-16s %d keypoints match' % (label, len(max_matches)))
                template_kp, template_des = self.templates_features[label][max_idx]

                # Find the homography matrix
                template_pts = np.float32([template_kp[m.queryIdx].pt for m in max_matches]).reshape(-1, 1, 2)
                query_pts = np.float32([query_kp[m.trainIdx].pt for m in max_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(template_pts, query_pts, cv2.RANSAC, 5.0)

                if M is None:
                    if self.print_message: print('%-16s Not enought inliers to calculate homography matrix' % '')
                    continue

                # Transform object keypoints to scene coordinates
                template_pts = np.float32([kp.pt for kp in template_kp]).reshape(-1, 1, 2)
                query_pts = cv2.perspectiveTransform(template_pts, M)

                # Find the bounding box coordinates
                x, y, w, h = cv2.boundingRect(query_pts)
                prediction[label] = [(x, y), (x + w, y + h)]
            else:
                if self.print_message: print('%-16s Not enough keypoints match are found - %d/%d' % (label, len(max_matches), self.min_match_thresh[label]))

        return prediction

    def show_message(self, flag: bool):
        self.print_message = flag
