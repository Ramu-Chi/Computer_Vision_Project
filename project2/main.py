import colorsys
import cv2
import numpy as np

from model import Model

template_root_path = 'project2/image/template'
query_file_path = 'project2/image/query/scene4.jpg'

# Get n distinct colors for each object bounding box
def get_label_colors(labels):
    def hsv2rgb(h, s, v): 
        (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
        return (int(255*r), int(255*g), int(255*b)) 

    huePartition = 1.0 / (len(labels) + 1) 
    object_box_colors = [hsv2rgb(huePartition * value, 1.0, 1.0) for value in range(len(labels))]
    return dict(zip(labels, object_box_colors))

if __name__ == '__main__':
    model = Model()
    model.show_message(True)
    model.load_templates(template_root_path)
    label_colors = get_label_colors(model.labels)

    query_img = cv2.imread(query_file_path)
    query_gray_img = cv2.imread(query_file_path, cv2.IMREAD_GRAYSCALE)

    prediciton = model.predict(query_gray_img)
    for label in prediciton.keys():
        bounding_box = prediciton[label]
        if bounding_box is None: continue

        x, y = bounding_box[0]
        cv2.rectangle(query_img, bounding_box[0], bounding_box[1], label_colors[label], 6)
        cv2.putText(query_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 3, label_colors[label], 6)

    scale = 1/4
    query_img = cv2.resize(query_img, None, fx=scale, fy=scale)
    cv2.imshow("SIFT Detector", query_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
