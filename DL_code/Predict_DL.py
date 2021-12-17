# coding=utf-8
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image as pil_image
from numpy import expand_dims

import colorsys
import random

from matplotlib import pyplot
from matplotlib.patches import Rectangle

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w, nb_box, scales_x_y):
    grid_h, grid_w = netout.shape[:2]
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5  # 5 = bx,by,bh,bw,pc

    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])  # x, y

    netout[..., :2] = netout[..., :2] * scales_x_y - 0.5 * (scales_x_y - 1.0)  # scale x, y

    netout[..., 4:] = _sigmoid(netout[..., 4:])  # objectness + classes probabilities

    for i in range(grid_h * grid_w):

        row = i / grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness
            objectness = netout[int(row)][int(col)][b][4]

            if (objectness > obj_thresh):
                # print("objectness: ", objectness)     #顯示找出所有bbox 分數

                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]

                x = (col + x) / grid_w  # center position, unit: image width
                y = (row + y) / grid_h  # center position, unit: image height

                w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height



                # last elements are class probabilities
                classes = objectness * netout[int(row)][col][b][5:]
                classes *= classes > obj_thresh
                box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
                boxes.append(box)
    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0]), int(x[1]), int(x[2])), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors
# get all of the results above a threshold
def get_boxes(boxes, labels, thresh, colors):
    v_boxes, v_labels, v_scores, v_colors = list(), list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):

            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
                v_colors.append(colors[i])
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores, v_colors
# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores, v_colors):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color=v_colors[i])
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')
    # show the plot
    pyplot.show()

def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, interpolation='bilinear', target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0

    # add a dimension so that we have one sample
    image = expand_dims(image, 0)

    return image, width, height

def load_image_pixels_flash(image_file, shape):
    image = pil_image.open(image_file)
    width, height = image.size      # 原始尺寸
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(shape, pil_image.BILINEAR)
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0
    image = expand_dims(image, 0)

    return image, width, height

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def load_model(GRAPH_PB_PATH, return_elements):
    tf.reset_default_graph()
    sess = tf.compat.v1.Session()
    print("load model graph")
    f = tf.io.gfile.GFile(GRAPH_PB_PATH, 'rb')
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    return_tensors = tf.import_graph_def(graph_def, return_elements=return_elements)
    print("Load Model Graph is DONE!")

    return sess, return_tensors

def process_bbox(pred_box, original_image_size, classes_path, anchors=[[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]], obj_thresh=0.25, input_size=608,scales_x_y=[1.2, 1.1, 1.05], nms_thresh= 0.5, class_threshold=0.6):
    boxes = list()
    for i in range(len(pred_box)):
        boxes += decode_netout(pred_box[i][0], anchors[i], obj_thresh, input_size, input_size, 3, scales_x_y[i])

    # Correct the boxes according the inital size of the image
    correct_yolo_boxes(boxes, original_image_size[1], original_image_size[0], input_size, input_size)

    # Suppress the non Maximal boxes
    do_nms(boxes, nms_thresh)
    # print("nb boxes remaining; ", len(boxes))

    # Get the details of the detected objects for a threshold > 0.6
    labels = get_classes(classes_path)
    colors = generate_colors(labels)
    v_boxes, v_labels, v_scores, v_colors = get_boxes(boxes, labels, class_threshold, colors)

    # Draw the result
    result = []
    for i in range(len(v_boxes)):
        result.append([v_labels[i], v_scores[i], v_boxes[i].xmin, v_boxes[i].ymin, (v_boxes[i].xmax - v_boxes[i].xmin), (v_boxes[i].ymax - v_boxes[i].ymin)])
    return result

# ======================================================================================================================

if __name__ == "__main__" :

    return_elements = ["input_1:0", "P5_out/BiasAdd:0", "P4_out/BiasAdd:0", "P3_out/BiasAdd:0"]

    GRAPH_PB_PATH = r'D:\Label_GVI_image\1101001_train\20211001\keras_model.pb'
    classes_path = r'D:\Label_GVI_image\1101001_train/KN.txt'

    image_path = r"D:\Label_GVI_image\B0083\W070D6101F_357.bmp"
    image_path = r"D:\Label_GVI_image\B0083\W06F9EC0A0_351.bmp"

    # init_參數
    input_size = 608  # 要為32倍數
    anchors = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]
    scales_x_y = [1.2, 1.1, 1.05]
    obj_thresh = 0.25   # 物件辨識閥值
    nms_thresh = 0.5    # BBOX合併閥值
    class_threshold = 0.6   # 類別閥值

    #----------------------------------------------------------------------------------------------
    # 先讀取model/graph方法
    sess, return_tensors = load_model(GRAPH_PB_PATH, return_elements)


    original_image = cv2.imread(image_path)
    original_image_size = original_image.shape[:2]

    image, image_w, image_h = load_image_pixels(image_path, (input_size, input_size))

    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
        feed_dict={return_tensors[0]: image})

    pred_box = [pred_lbbox, pred_mbbox, pred_sbbox]

    # =============================================================================================
    result = process_bbox(pred_box, original_image_size)



