# coding=utf-8
import base64
import json
import time

import requests


api = 'http://172.26.99.140:8080/test'
image_file = r"D:/Label_GVI_image/B0083/W06F9EC0A1_302.bmp"

name = image_file.rsplit('/', 1)[1]

start_time = time.time()

with open(image_file, "rb") as f:
    im_bytes = f.read()
im_b64 = base64.b64encode(im_bytes).decode("utf8")

headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'type': image_file.rsplit('.', 1)[1], 'filename': name}
payload = json.dumps({"image": im_b64, "other_key": "value"})
response = requests.post(api, data=payload, headers=headers)
end_time = time.time()
try:
    data = response.json()
    print(data)
except requests.exceptions.RequestException:
    print(response.text)


print("cost time : ", end_time-start_time)