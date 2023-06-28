### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import torch
from flask import Flask, request, jsonify, send_file
import imageio
import cv2
from datetime import datetime
from mtcnn import MTCNN
from PIL import Image
import replicate
from io import BytesIO
import numpy as np
from Utils.VideoAgingUtils import VideoAgingUtils
from Utils.GCPUtils import GCPUtils

#Set the REPLICATE_API_TOKEN environment variable
os.environ["REPLICATE_API_TOKEN"] =  "r8_W9eZaZk2Xdikeun3JeH0GZC97kLuZyY16Kkob"

app = Flask(__name__)

@app.route('/predict-using-sam-model', methods=['POST'])
def predict_using_sam_model():
    print("[START] Predict with SAM model")
    input_image = request.files['file']

    # Đọc dữ liệu từ input_image
    image_data = input_image.read()
   
    output = replicate.run(
        "yuval-alaluf/sam:9222a21c181b707209ef12b5e0d7e94c994b58f01c7b2fec075d2e892362f13c",
        input={"image": BytesIO(image_data), "target_age": "default"}
    )
    # output_file_path = save_image_to_gcloud(gif_path)

    data = {
        'outputFilePath': output
    }
    print("[END] Predict with SAM model")
    return jsonify(data)

@app.route('/predict-using-sam-model-with-target-age', methods=['POST'])
def predict_using_sam_model_with_target_age():
    print("[START] Predict with SAM model with target age")
    input_image = request.files['file']
    try:
        target_age = request.form['targetAge']
        print('Target age = ' + target_age)
    except (KeyError, ValueError, TypeError):
        # Handle the case when the parameter is missing or not a valid integer
        return "Invalid parameter: 'targetAge' must be provided as an integer", 400

    # Đọc dữ liệu từ input_image
    image_data = input_image.read()
   
    output = replicate.run(
        "yuval-alaluf/sam:9222a21c181b707209ef12b5e0d7e94c994b58f01c7b2fec075d2e892362f13c",
        input={"image": BytesIO(image_data), "target_age": str(target_age)}
    )
    # output_file_path = save_image_to_gcloud(gif_path)

    data = {
        'outputFilePath': output
    }
    print("[END] Predict with SAM model with target age")
    return jsonify(data)



@app.route('/extract-portrait', methods=['POST'])
def extract_portrait():
    print("[START] Extract portrait")
    # Check if the 'file' key is in the request files
    if 'video_file' not in request.files:
        return "No file uploaded", 400

    video_file = request.files['video_file']

    # Check if the file is a video
    if video_file.filename == '':
        return "Invalid file", 400

    # Đọc video từ file tạm thời
    temp_filename = 'temp_video.mp4'
    video_file.save(temp_filename)

    # Open the video stream using cv2.VideoCapture
    video = cv2.VideoCapture(temp_filename)

    # Load pre-trained face detection model (MTCNN)
    detector = MTCNN()

    # Read the first frame
    success, frame = video.read()

    while success:
        # Đọc khung hình tiếp theo
        success, frame  = video.read()
        # Detect faces in the frame
        results = detector.detect_faces(frame)
        print(f"Number face have detected: {len(results)}")
        if len(results) > 0:
            # Get the first face (assuming it's the person of interest)
            face = results[0]['box']
            x, y, w, h = face

            # Calculate the expanded region around the face
            padding = int(min(w, h))
            x -= padding
            y -= padding
            w += padding
            h += padding

            # Ensure the expanded region is within the frame boundaries
            x = max(0, x)
            y = max(0, y)
            x_end = min(frame.shape[1], x + w)
            y_end = min(frame.shape[0], y + h)

            # Calculate the center coordinates of the face
            face_center_x = (x + x_end) // 2
            face_center_y = (y + y_end) // 2

            # Calculate the top-left coordinates for the cropped region
            crop_x = max(0, face_center_x - int((w + 2 * padding) / 2))
            crop_y = max(0, face_center_y - int((h + 2 * padding) / 2))

            # Calculate the size of the cropped region
            crop_size = min(frame.shape[0] - crop_y, frame.shape[1] - crop_x)

            # Extract the portrait region from the frame
            portrait = frame[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

            # Tạo một đối tượng BytesIO và ghi portrait vào đó
            # Convert the portrait to a NumPy array
            portrait_np = np.array(portrait)

            # Encode the image as JPEG
            _, jpeg_data = cv2.imencode('.jpg', portrait_np)

            # Create a BytesIO object and write the JPEG data to it
            output_stream = BytesIO(jpeg_data.tobytes())

            # Tạo một đối tượng BytesIO và ghi portrait vào đó
            portrait_url = GCPUtils().save_portrait_to_gcs(output_stream)

            data = {
                'portrait': portrait_url
            }

            # Trả về portrait dưới dạng multipart file
            return jsonify(data)

    # Release the video capture object
    video.release()
    os.remove(temp_filename)

    data = {
        'portrait': 'File not found'
    }
    print("[END] Extract portrait")
    return jsonify(data)

@app.route('/aging-video', methods=['POST'])
def aging_video():
    print("[START] Aging Video")
    video_utils = VideoAgingUtils()
    # Check if the 'file' key is in the request files
    if 'video_file' not in request.files:
        return "No file uploaded", 400

    video_file = request.files['video_file']

    # Check if the file is a video
    if video_file.filename == '':
        return "Invalid file", 400

    # Đọc video từ file tạm thời
    temp_filename = 'video/temp_video.mp4'
    video_file.save(temp_filename)
    
    fps = 30
    output_video_path = "video/output_video.mp4"

    # Open the video stream using cv2.VideoCapture
    video = cv2.VideoCapture(temp_filename)

    # Đếm số lượng khung hình
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    success, image = video.read()
    height, width, _ = image.shape

    # Tạo đối tượng VideoWriter để ghi video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Chọn mã hóa video (trong ví dụ này, sử dụng mp4v)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Đọc ảnh khuôn mặt
    # face = cv2.imread(face_path)
    aged_image = request.files['aged_image']
    face_data = aged_image.read()
    nparr = np.fromstring(face_data, np.uint8)
    face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    count = 0

    # Trích xuất các khung hình
    while success:
        count += 1
        print(f"[Aging] Frame {count}/{frame_count}")
        try:
            # Đọc khung hình tiếp theo
            success, body = video.read()

            swap_face_image = video_utils.swap_face(video_utils, face, body)
            # swap_face_images.append(swap_face_image)

            video_writer.write(swap_face_image)
        except Exception as e:
            # Handling other exceptions
            print("An error occurred:", str(e))

    output_gcp_path = GCPUtils().upload_video(output_video_path)
    
    # Đóng video
    video.release()
    video_writer.release()

    data = {
        'outputFilePath': output_gcp_path
    }
    print("[END] Aging Video")
    return jsonify(data)
    
if __name__ == '__main__':
    app.run()