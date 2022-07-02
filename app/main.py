from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, Body
import numpy as np
from starlette.requests import Request
from fastapi.responses import JSONResponse, Response
from fastapi.encoders import jsonable_encoder
import os
import cv2
import time
import datetime
import json
import base64
import io
from urllib.request import urlopen
from PIL import Image, ImageDraw, ImageColor, ImageFont
import boto3
import pandas as pd
import requests
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import GenericMask
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import pickle5 as p

s3 = boto3.client('s3')


app = FastAPI()


cfg_save_path = "IS_cfg.pickle"


with open(cfg_save_path, 'rb') as f:
    cfg = p.load(f)
    
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") #load model weight path of custom dataset we trained

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
predictor = DefaultPredictor(cfg)

def on_image(img_np):
    im = img_np
    outputs = predictor(im)
    h = im.shape[0]
    w = im.shape[1]
    mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
    masks = [GenericMask(x, h, w) for x in mask_array]
    for m_i in range(len(masks)):
        mask_curr_polygons = [a.astype(int) for a in masks[m_i].polygons]
        for ix in range(len(mask_curr_polygons)):
            x, y = mask_curr_polygons[ix][::2], mask_curr_polygons[ix][1::2]
    print(x[0])
    mask_co = []
    for i in range(len(x)):
        new_x = x[i]
        new_y = y[i]
        ele = [str(new_x), str(new_y)]
        mask_co.append(ele)
    
    num_instances = mask_array.shape[0]
    num_instance = mask_array.shape
    print('num_instance : ',num_instance)
    scores = outputs['instances'].scores.to("cpu").numpy()
    labels = outputs['instances'].pred_classes .to("cpu").numpy()
    bbox   = outputs['instances'].pred_boxes.to("cpu").tensor.numpy()

    mask_array = np.moveaxis(mask_array, 0, -1)

    mask_array_instance = []
    #img = np.zeros_like(im) #black
    h = im.shape[0]
    w = im.shape[1]
    color = (200, 100, 255)
    v = Visualizer(im[:,:,::-1], metadata={}, scale= 0.4, instance_mode = ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #output_dir = 'output_test.jpg'
    #cv2.imwrite(output_dir, v.get_image())
    return v.get_image(), mask_co
    
 

                
@app.post("/text_on_image")
def image_gen(request: Request, userPhoto: Optional[bytes] = File(None), url: Optional[str] = Body(None) ):
    if url is not None: 
        image_r = requests.get(url)
        fetch_status = image_r.status_code
        if fetch_status == 200:
            image = image_r.content
            img_np = cv2.imdecode(np.asarray(bytearray(image), dtype=np.uint8), 1)
    elif userPhoto is not None:
        nparr = np.fromstring(userPhoto, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        return {"response":"Please provide url or image"}
    #text = 'hey'
    s3_push_img, dict_output = on_image(img_np)
    cv2.imwrite("output.jpg", s3_push_img)
    #dict_output = str(dict_output)
    return dict_output
if __name__ == '__main__':
    unvicorn.run(app, host = '121.0.0.1', port=8001)

    
    
    
    
    

    
   


