import shutil
import time
import io
from fastapi.testclient import TestClient
from app.main import app, BASE_DIR, UPLOAD_DIR

from PIL import Image, ImageChops

img_saved_path = BASE_DIR / "images"

client = TestClient(app)

def test_prediction_upload():
    img_saved_path = BASE_DIR / "images"
    for path in img_saved_path.glob("*"):
        print(path)
        try:
            img = Image.open(path)
        except:
            img = None
        response = client.post("/",
            files={"file": open(path, 'rb')}
            #headers={"Authorization": f"JWT {settings.app_auth_token}"}
        )
        if img is None:
            assert response.status_code == 400
        else:
            # Returning a valid image
            assert response.status_code == 200
            data = response.json()
            assert len(data.keys()) == 3

def test_echo_upload():
    img_saved_path = BASE_DIR / "images"
    for path in img_saved_path.glob("*"):
        try:
            img = Image.open(path)
        except:
            img = None
        response = client.post("/img-echo/", files={"file": open(path, 'rb')})
        if img is None:
            assert response.status_code == 400
        else:
            # Returning a valid image
            assert response.status_code == 200
            r_stream = io.BytesIO(response.content)
            echo_img = Image.open(r_stream)
            difference = ImageChops.difference(echo_img, img).getbbox()
            #assert difference is None
    time.sleep(10)
    shutil.rmtree(UPLOAD_DIR)