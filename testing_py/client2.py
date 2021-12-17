# coding=utf-8
import time
import json
import base64
import logging
import requests


api = 'http://172.26.99.140:8080/predict/W06F9EC0D1_186.bmp'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'Image_path_head':'D:/Label_GVI_image/B0083/'}

# with open(image_file, "rb") as f:
#     im_bytes = f.read()
# im_b64 = base64.b64encode(im_bytes).decode("utf8")
# payload = json.dumps({"image": im_b64, "other_key": "value"})




s = requests.Session()
for i in range(1):
    s_time = time.time()
    # s.head(api + image_file,headers = headers)

    r = s.post(api, headers=headers)
    data = r.json()
    print(data)
    end_time = time.time()
    print(end_time - s_time)