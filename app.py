from flask import Flask, Response,render_template,request,url_for,flash
import cv2
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import atexit, shutil

# mentioning variables
app = Flask(__name__)
app.secret_key='kirti'
resized=(488,368)
model = load_model('pcos_model_vgg.h5')  # Load the trained model
global capture,switch
capture=0
switch=0
camera=cv2.VideoCapture(0)
predicted_class=''
what=''
why=''
cure=''

@app.route('/')
def index():
    image_url = url_for('static', filename='temp.jpg')
    return render_template('index1.html', image_url=image_url)

# uploading from folder
@app.route('/file_upload', methods=['POST'])
def file_upload():
    print("request received")
    image_data = request.files['image']  # Access the uploaded file
    image_path = 'temp.jpg'  # Save the uploaded file to a temporary location
    print("trying save image")
    image_data.save(image_path) 
    shutil.copyfile(image_path,'./static/temp.jpg')
    print("image saved")
    os.remove(image_path)
    return render_template("index1.html")

# uploading from camera
def gen_frames():
    global capture, camera

    while True:
        if capture:
            capture = False
            image_path = "./static/temp.jpg"

            # Reading a frame from the camera
            success, frame = camera.read()
            if success:
                try:
                    # Fliping the frame horizontally
                    frame = cv2.flip(frame, 1)
                    # Saving the captured frame as an image
                    # print("saving...")
                    cv2.imwrite(image_path, frame)
                    # print("saved.")
                    # Processing the frame and yielding it as a response
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    pass

            # Releasing the camera and breaking the loop
            try:
                camera.release()
            except AttributeError:
                pass    
            break

        # Continuing reading frames from the camera
        success, frame = camera.read()
        if success:
            try:
                # Process the frame and yield it as a response
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

    # camera resources clean up
    cv2.destroyAllWindows()
    camera = None

@app.route('/video_feed')   #video frame
def video_feed():
    global switch
    if switch == 1:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response('', mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global camera,switch
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('stop') == 'Turn on Camera':
            camera = cv2.VideoCapture(0)
            switch=1 
                          
    return render_template("index1.html")
# uploading from camera ends

# prediction and result
@app.route('/predict',methods=['POST','GET'])
def predict():
    image_path=''
    if(os.path.exists('./static/temp.jpg')):
        image_path="./static/temp.jpg"
    else:
        flash("No image uploaded!")
        return render_template("index1.html")
    
    image = load_img(image_path, target_size=(100, 100))
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # Normalize the image array
    image_array = np.expand_dims(image_array, axis=0)  # Add an extra dimension

    prediction = model.predict(image_array)[0]
    class_labels = ['AN','Non_pcos','acne','facial_hair']
    predicted_class = class_labels[np.argmax(prediction)]
    
    image_path=str(image_path)
    # Print the predicted class in the terminal
    print('Predicted class:', predicted_class)
    print(image_path)
    # Remove the temporary image file
    # os.remove(image_path)
    return render_template("index2.html",predicted_class=predicted_class,image_source=image_path)


# Cleanup function to remove the temp files/images
def cleanup():
    # Delete the files
    file_paths = ['./static/temp.jpg', 'temp.jpg']
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass    

# Registering the cleanup function to run when the Flask application is terminated
atexit.register(cleanup)           


if __name__ == '__main__':
    app.debug=True
    app.run()
       
