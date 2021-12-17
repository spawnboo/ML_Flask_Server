# coding=utf-8
import json
import urllib3
import base64
import time


api = 'http://localhost:80/test'
image_file = r"D:/Label_GVI_image/B0083/W06F9EC0A1_302.bmp"
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
with open(image_file, "rb") as f:
    im_bytes = f.read()
im_b64 = base64.b64encode(im_bytes).decode("utf8")
payload = json.dumps({"image": im_b64, "other_key": "value"})



encoded_body = json.dumps({
        "description": "Tenaris",
        "ticker": "TS.BA",
        "industry": "Metalúrgica",
        "currency": "ARS",
    })

http = urllib3.PoolManager()

for i in range(5):
    s_time = time.time()
    r = http.request('POST', api,
                     headers={'Content-Type': 'application/json'},
                     body=payload)
    data = r.data
    values = json.loads(data)
    print(values)


    end_time = time.time()
    print(end_time - s_time)
