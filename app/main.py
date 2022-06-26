import pathlib
import io
from functools import lru_cache
import uuid
import numpy as np
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Request,
    File,
    UploadFile
    )
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseSettings
from PIL import Image
from app.model import create_trained_model, make_prediction
import base64

import pickle as pkl

#https://www.pluralsight.com/tech-blog/porting-flask-to-fastapi-for-ml-model-serving/

BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "results"

MODEL_DIR = BASE_DIR.parent / "models"
YOLO_MODEL_PATH_DIR = MODEL_DIR / "yolov3"
MODEL_PATH = YOLO_MODEL_PATH_DIR / "best_weights_final_18.hdf5"
MODEL = create_trained_model(MODEL_PATH)

app = FastAPI()


#Rest API
@app.post("/prediction/") #http POST
async def prediction_view(file:UploadFile = File(...), authorization = Header(None)):
    #verify_auth(authorization, settings)
    bytes_str =  io.BytesIO(await file.read()) #read image as a byte stream
    try:
        img = Image.open(bytes_str)
    except:
        raise HTTPException(detail="Invalid image", status_code=400)
    print(img)

    pkl.dump(img, open('/tmp/image', 'wb'))

    image = np.array(img)
    fname = pathlib.Path(file.filename)
    fext = fname.suffix or '.jpg'
    print(fext)

    UPLOAD_DIR.mkdir(exist_ok=True)

    predictions = make_prediction(MODEL, image, fext)
    encoded_img = predictions['encoded_img']

    decoded_img = base64.b64decode(encoded_img)
    dest = UPLOAD_DIR / fname

    with open(dest, 'wb') as f_output:
        f_output.write(decoded_img)

    return {"results": predictions['data'], "image_encoded": predictions['encoded_img'], "original": predictions}

@app.post("/img-echo/", response_class=FileResponse) #http post
async def img_echo_view(file:UploadFile = File(...)):
    #if not settings.echo_active:
    #    raise HTTPException(detail="Invalid endpoint", status_code=400)
    UPLOAD_DIR.mkdir(exist_ok=True)
    bytes_str =  io.BytesIO(await file.read()) #read image as a byte stream
    try:
        img = Image.open(bytes_str)
    except:
        raise HTTPException(detail="Invalid image", status_code=400)
    fname = pathlib.Path(file.filename)
    fext = fname.suffix #.jpg
    #use uuid1 which contaings a timestamp for file naming
    dest = UPLOAD_DIR / f"{uuid.uuid1()}{fext}"
    img.save(dest)
    return dest

