import os
from flask import Flask, request, redirect, render_template, flash, send_file, make_response, Response
from werkzeug.utils import secure_filename
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
# app.config['JSON_AS_ASCII'] = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#学習済みモデルをロード
model = YOLO('./best.pt')
state_dict = model.state_dict()
saved_state_dict = torch.load("./model_para.pt")
for name, value in saved_state_dict.items():
    state_dict[name] = value
model.load_state_dict(state_dict, strict=False)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # 受け取った画像をモデルに渡して推論する
            with torch.no_grad():
                results = model.predict(filepath, save=False)
                
            img=cv2.imread(filepath)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            if len(results[0].boxes.xyxy):
                boxes=results[0].boxes.xyxy.cpu().numpy()
                scores=results[0].boxes.conf.cpu().numpy()
                show_bbox_tmp(img,boxes,scores)
  
            # レスポンスに画像を添付する
            # 画像をメモリ上に保存する
            image = io.BytesIO()
            Image.fromarray(img).save(image, 'PNG')
            image.seek(0)
            # レスポンスオブジェクトを作成する
            response = make_response(send_file(image, mimetype='image/png'))
            return response
    return render_template("index.html",answer="")

def show_bbox_tmp(img,boxes,scores,color=(0,255,0)):
    boxes=boxes.astype(int)
    scores=scores
    for i,box in enumerate(boxes):
        score=f"{scores[i]:.4f}"
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color,2)
        y=box[1]-10 if box[1]-10>10 else box[1]+10
        cv2.putText(img,score,(box[0],y),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

# if __name__ == "__main__":
#      app.run()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)