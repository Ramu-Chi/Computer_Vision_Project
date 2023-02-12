import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import xml.etree.ElementTree as ET

from model import Model

template_root_path = 'project2/image/template'
test_root_path = 'project2/image/query'
labels = os.listdir(template_root_path)

def parse_xml(manual_label_file):
    tree = ET.parse(manual_label_file)
    root = tree.getroot()

    ground_truth = dict.fromkeys(labels)
    for object in root.iter('object'):
        label = object.find('name').text
        bndbox = object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        ground_truth[label] = [(xmin, ymin), (xmax, ymax)]
    return ground_truth

def evaluate_model(model: Model, draw_plot=False):
    iou_thresh_list = [0.05 * i for i in range(10, 19)]
    f1_scores_testset = []

    for img_file in os.listdir(test_root_path):
        if not ( img_file.endswith('png') or img_file.endswith('jpg') ): continue
        label_file = img_file[:-3] + 'xml'
        test_img = cv2.imread(os.path.join(test_root_path, img_file), cv2.IMREAD_GRAYSCALE)

        if not os.path.isfile(os.path.join(test_root_path, label_file)): continue
        ground_truth = parse_xml(os.path.join(test_root_path, label_file))
        prediction = model.predict(test_img)
        
        f1_scores_img = []
        for iou_thresh in iou_thresh_list:
            f1_scores_img.append(get_f1_score(ground_truth, prediction, iou_thresh))

        f1_scores_testset.append(f1_scores_img)

    f1_scores_testset = np.mean(f1_scores_testset, axis=0)

    if draw_plot:
        plt.plot(iou_thresh_list, f1_scores_testset)
        plt.xlabel('IoU Threshold')
        plt.xticks(iou_thresh_list)
        plt.ylabel('F1 Score')
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.title('F1 Score vs IoU Threshold')
        plt.savefig('test.png')
        plt.show()
    return np.mean(f1_scores_testset)

def get_f1_score(ground_truth, prediction, iou_threshold):
    TP = 0
    for label in labels:
        if ground_truth[label] is None or prediction[label] is None: continue
        if intersection_over_union(ground_truth[label], prediction[label]) >= iou_threshold:
            TP += 1
    
    try:
        precision = TP / len([i for i in prediction.values() if i != None])
        recall = TP / len([i for i in ground_truth.values() if i != None])
        return 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if TP == 0 and not any(prediction.values()) and not any(ground_truth.values()): # TP = FP = FN = 0
            return 1
        return 0

def intersection_over_union(box1, box2):
    if box1 is None or box2 is None: return 0
    
    intersect_box = [np.maximum(box1[0], box2[0]), np.minimum(box1[1], box2[1])]
    i_xmin, i_ymin = intersect_box[0]
    i_xmax, i_ymax = intersect_box[1]

    i_area = max(0, i_xmax - i_xmin) * max(0, i_ymax - i_ymin)
    box1_area = np.product(np.subtract(box1[1], box1[0]))
    box2_area = np.product(np.subtract(box2[1], box2[0]))

    iou = i_area / (box1_area + box2_area - i_area)
    return iou

model = Model()
model.LOWE_RATIO = 0.6
model.sift = cv2.SIFT_create(contrastThreshold=0.05, edgeThreshold=80)
model.load_templates(template_root_path)
score = evaluate_model(model, draw_plot=True)
print('Average F1 Score over IoU thresholds from 0.5 to 0.9:', score)


''' 
Evaluate influence of Contrast Threshold in SIFT 
uncomment and run (~ 12mins)
'''
# contrast_thresh_list = [i * 0.01 for i in range(1, 13)]
# scores = []

# for i, contrast_thresh in enumerate(contrast_thresh_list):
#     print(f'{i + 1}/{len(contrast_thresh_list)}')
#     model = Model()
#     model.sift = cv2.SIFT_create(contrastThreshold=contrast_thresh, edgeThreshold=30)
#     model.load_templates(template_root_path)

#     score = evaluate_model(model)
#     scores.append(score)

# plt.plot(contrast_thresh_list, scores)
# plt.xlabel('Contrast Threshold')
# plt.xticks(contrast_thresh_list)
# plt.ylabel('Mean F1 Score Over IoU Thresholds')
# plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# plt.title('Evaluation of contrast threshold of SIFT')
# plt.savefig('contrast_threshold.png')
# plt.show()


''' 
Evaluate influence of Edge Threshold in SIFT
uncomment and run (~ 12mins)
'''
# edge_thresh_list = [i * 10 for i in range(1, 13)]
# scores = []

# for i, edge_thresh in enumerate(edge_thresh_list):
#     print(f'{i + 1}/{len(edge_thresh_list)}')
#     model = Model()
#     model.sift = cv2.SIFT_create(contrastThreshold=0.05, edgeThreshold=edge_thresh)
#     model.load_templates(template_root_path)

#     score = evaluate_model(model)
#     scores.append(score)

# plt.plot(edge_thresh_list, scores)
# plt.xlabel('Edge Threshold')
# plt.xticks(edge_thresh_list)
# plt.ylabel('Mean F1 Score Over IoU Thresholds')
# plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# plt.title('Evaluation of edge threshold of SIFT')
# plt.savefig('edge_threshold.png')
# plt.show()


''' 
Evaluation of Lowe ratio
uncomment and run (~ 8mins)
'''
# model = Model()
# model.sift = cv2.SIFT_create(contrastThreshold=0.05, edgeThreshold=80)
# model.load_templates(template_root_path)

# lowe_ratio_list = [0.05 * i for i in range(4, 19)]
# scores = []

# for i, lowe_ratio in enumerate(lowe_ratio_list):
#     model.LOWE_RATIO = lowe_ratio
#     score = evaluate_model(model)
#     scores.append(score)

# plt.plot(lowe_ratio_list, scores)
# plt.xlabel('Lowe Ratio')
# plt.xticks(lowe_ratio_list)
# plt.ylabel('Mean F1 Score Over IoU Thresholds')
# plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# plt.title('Evaluation of Lowe ratio test')
# plt.savefig('lowe_ratio.png')
# plt.show()
