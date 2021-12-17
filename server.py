# coding=utf-8
import os
import io
import base64
import logging
from flask import Flask, request, jsonify
from DL_code import Predict_DL as DL
import time

# 上傳位置
UPLOAD_FOLDER = r'D:\PycharmProjects\Flask_docker\temp'

# Model 於本機位置
Temp_Model_path = r"D:\DL_Model"

# 起動Flask
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

sess,return_tensors  = -1, []       # 沒有值的model 參數

# DL init
anchors = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]
obj_thresh = 0.25
input_size = 608  # 要為32倍數
scales_x_y = [1.2, 1.1, 1.05]
nms_thresh = 0.5
class_threshold = 0.6

# 查看目前設定參數
@app.route(("/init_Check"),methods=['GET'])
def init_check_method():
    init_value = {}
    init_value['anchors'] = anchors
    init_value['obj_thresh'] = obj_thresh
    init_value['input_size'] = input_size
    init_value['scales_x_y'] = scales_x_y
    init_value['nms_thresh'] = nms_thresh
    init_value['class_threshold'] = class_threshold
    return jsonify(init_value)

# /modelpath 顯示目前model 所有路徑與方法
@app.route("/modelpath", methods=['GET'])
def modelpath_method():
    back_path = []
    for root, dirs, files in os.walk(Temp_Model_path):
        print("Path：", root)
        back_path.append(root)

    return jsonify(back_path)

# /test 頁籤
@app.route("/test", methods=['POST'])
def test_method():
    start_time = time.time()
    # 我猜應該是, 如果上傳結果非 影像或json 則 顯示錯誤(400)
    # if not request.json or 'image' not in request.json:
    #     abort(400)

    # get the base64 encoded string
    im_b64 = request.json['image']
    # convert it into bytes
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    image, width, height = DL.load_image_pixels_flash(io.BytesIO(img_bytes), (input_size, input_size))
    original_image_size = [width, height]      # 後面用的到 原始尺寸大小

    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
        feed_dict={return_tensors[0]: image})

    pred_box = [pred_lbbox, pred_mbbox, pred_sbbox]
    # 處理BBOX 到結果框
    result = DL.process_bbox(pred_box, original_image_size, classes_path)

    # for head in request.headers:
    #     print(head)


    # process your img_arr here

    # access other keys of json
    # print(request.json['other_key'])

    result_dict = {'output': 'output_key'}
    end_time = time.time()

    result.append([str(end_time-start_time)])

    return jsonify(result)
    # return result_dict

# /model 載入Model 的方法
@app.route("/model", methods=['POST'])
def load_model_method():
    global model_path_head, classes_path
    global sess, return_tensors

    model_path_head = request.headers['model_name_head']  # 擷取model_path_head 的 header
    return_elements = ["input_1:0", "P5_out/BiasAdd:0", "P4_out/BiasAdd:0", "P3_out/BiasAdd:0"]

    # 檢查檔案是否存在
    model_path = os.path.join(Temp_Model_path, model_path_head, model_path_head+".pb")
    if os.path.isfile(model_path):
        sess, return_tensors = DL.load_model(os.path.join(model_path, ), return_elements)
    else:
        time.sleep(0.1) # 不加sleep 有時會傳不回訊號
        return jsonify({"model Load False!!! Model_Path": str(model_path)})

    classes_path_check = os.path.join(Temp_Model_path, model_path_head, "defectCodes.txt")
    if os.path.isfile(classes_path_check):
        classes_path = classes_path_check
    else:
        time.sleep(0.1)  # 不加sleep 有時會傳不回訊號
        return jsonify({"Classes Load False!!! Classes_Path": str(classes_path_check)})


    return jsonify({"model Load Done! Model_Path":  model_path})

# /predict 圖片預測的方法
@app.route("/predict", methods=['POST'])
def predict_method():
    # Get all Header
    ALL_Headers = request.headers

    # if ALL_Headers.get('Image_path_head') == None:  # 找不到路徑位置, 請重新給予
    #    return jsonify('Not get "Image_path_head", Please check!!! ')

    if ALL_Headers.get('nms_threshold_head') != None:   # 有額外給予NMS 閥值, 重新定義
        if float(ALL_Headers['nms_threshold_head']) <1 and float(ALL_Headers['nms_threshold_head']) > 0:
            global nms_thresh
            nms_thresh = float(ALL_Headers['nms_threshold_head'])

    if ALL_Headers.get('classes_threshold_head') != None:   # 有額外給予Classes 閥值, 重新定義
        if float(ALL_Headers['classes_threshold_head']) <1 and float(ALL_Headers['classes_threshold_head']) > 0:
            global class_threshold
            class_threshold = float(ALL_Headers['classes_threshold_head'])

    if sess == -1:  # 檢查Model 是否有載入
        return jsonify({"Model is not Set, please Load Model!!!"})

    # -------------------------------------------------------------------------------------------------------------------
    # image_path_head    = request.headers['Image_path_head']             # 擷取Image_path_head 的 header
    data = request.get_json()
    image_path_head = data['ImagePath']

    if os.path.isfile(image_path_head):             # 檢查是否有存在此檔案
        image, width, height = DL.load_image_pixels(os.path.join(image_path_head), (input_size, input_size))
        original_image_size = [width, height]      #後面用的到 原始尺寸大小

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
            feed_dict={return_tensors[0]: image})

        pred_box = [pred_lbbox, pred_mbbox, pred_sbbox]
        # 處理BBOX 到結果框
        result = DL.process_bbox(pred_box, original_image_size, classes_path, anchors, obj_thresh, input_size, scales_x_y, nms_thresh, class_threshold)

        return jsonify(result)
    else:
        return jsonify({"Image path not exist, path: ":  image_path_head})



def run_server_api():
    app.run(host='172.26.99.140', port=8080, threaded=True)


if __name__ == "__main__":
    run_server_api()