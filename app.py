from ultralytics import YOLO 
import easyocr
import cv2
from flask import Flask, render_template, request, redirect, url_for,jsonify
import os 
import time
import base64

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def index():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
    return render_template('index.html', filepath=filepath)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Get the uploaded image data
    image_data = request.form['image']

    # Decode the base64 image data
    img_bytes = base64.b64decode(image_data.split(',')[1])
    print("Image data decoded ------------", img_bytes)

    # Save the decoded image to a file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
    with open(filepath, 'wb') as f:
        f.write(img_bytes)

    # Perform processing on the image (replace this with your own processing logic)
    # For example, you can use OpenCV for image processing
    print("Image saved at", filepath)

    # Redirect to the result page
    # return render_template('index.html', filepath=filepath)
    return redirect('/detection')

@app.route('/delete_image', methods=['POST'])
def delete_image():
    if request.method == 'POST':
        image_name = request.form['image_name']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
        if os.path.exists(image_path):
            os.remove(image_path)
    return redirect(url_for('index'))

@app.route('/detection', methods=['POST'])
def upload_success():
    # file name is the file name that some one uploaded 
    # so the image path will be ./static/uploads/<filename>
    # call your model with keras tf of sklearn
    try :
        model = YOLO('./static/model/yolov8-custom.pt')
        confidence = 0.6 # 0.0-1.0    # render the message that you wanted to show in message variable
        results = model.predict(source = f"./static/uploads/input.jpg", save = True, show = False, conf = confidence)
        print(results)

        img_main = cv2.imread("./static/uploads/input.jpg")
        r = results[0]
        box = r.boxes[0]
        [left, top, right, bottom] = box.xyxy[0]
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        cropped_img = img_main[top+1:bottom-1, left+1:right-1]
        output_path = "./static/uploads/processed.jpg"
        cv2.imwrite(output_path, cropped_img)
        print(f"Largest image saved at {output_path}")

        time.sleep(2)
        print("Detection and cropping is successfull. ")
        
        return redirect('/easyocr_detection')
    
    except Exception as e :
        print(e)
        return render_template('error.html')




@app.route('/easyocr_detection')
def easyocr_detection():
    try:
        print("Now Detecting using Easy OCR.")
        reader = easyocr.Reader(['bn'], gpu=False)
        result = reader.readtext("./static/uploads/processed.jpg", detail=0, paragraph=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')  # file path of the image to be uploaded
        print(result)
        
        return render_template('index.html', result=result[0], filepath=filepath)
    
    except Exception as e:
        return render_template('error.html')

@app.route('/dmp_fix/')
def dmp_fix():
    try:
        print("Now easy ocr part.")
        reader = easyocr.Reader(['bn'], gpu = False)
        result = reader.readtext("./static/uploads/processed.jpg", detail = 0, paragraph = True)
        print(result)
        return jsonify(result)
    except Exception as e :
        return jsonify(e)

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000) # Run the server on a different port than Rasa's
