import cv2
import os
import imagehash
from PIL import Image
import numpy as np
import dlib
from scipy.spatial import distance as dist
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('assets/face_predictor.dat')
MIN_FACE_WIDTH = 80 #Face pix
MIN_FACE_HEIGHT = 80
def load_image(image_path):
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def load_yolo_model():
    try:
        yolo_net = cv2.dnn.readNet('assets/yolov3.weights', 'assets/yolov3.cfg')
        layer_names = yolo_net.getLayerNames()
        unconnected_out_layers = yolo_net.getUnconnectedOutLayers()
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
        
        return yolo_net, output_layers
    
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        raise

def detect_objects(image, yolo_net, output_layers):
    
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            if detection.ndim == 1:  # Handle case where detection is 1D array
                detection = detection.reshape(1, -1)
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    objects = [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []
    
    cropped_objects = [crop_object(image, *box) for box in objects]

    return cropped_objects

def crop_object(image, x, y, w, h):
    
    return image[y:y+h, x:x+w]
def get_blur(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurr = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return blurr

def quality(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurriness = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    exposure = gray_image.mean()
    noise = gray_image.std()
    
    return blurriness, exposure, noise

def get_criteria(reference_image):
    reference_image = load_image(reference_image)
    blurriness, exposure, noise = quality(reference_image)
    
    criteria = {
        'blurriness_max' : blurriness * 3.5,
        'exposure_min': exposure * 0.7,
        'exposure_max':exposure * 4,
        'noise_max': noise * 4,
    }
    
    print(f"Reference Criteria:{criteria}")
    
    return criteria 

def hash_image(image_path):
    
    hash_value = imagehash.phash(Image.open(image_path))
    #print(f"Generated Hash for {image_path}: {hash_value}")  # Debug
    return hash_value

def detect_duplicates(image_path,hashes, threshold):
    image_hash = hash_image(image_path)
    #print(f"Checking duplicates for {image_path} with hash {image_hash}")  # Debug
    is_duplicate = any(
        isinstance(existing_hash, imagehash.ImageHash) and
        (image_hash - existing_hash) < threshold # subtracts image_hash in all existing_hash
        for existing_hash in hashes.values()
    )
    #print(f"Is {image_path} a duplicate? {is_duplicate}")  # Debug
    return is_duplicate

LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))  
def resize_image(image, width=2000):
    """Resize the image to the specified width while maintaining aspect ratio."""
    aspect_ratio = width / float(image.shape[1])
    height = int(image.shape[0] * aspect_ratio)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image

def calculate_ear(eye):  #standard
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def are_eyes_closed(image, ear_threshold=0.2):
    """Detects if eyes are closed in a given image based on EAR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        return False  # No face detected

    for face in faces:
        # Check if the face is large enough to process
        if face.width() < MIN_FACE_WIDTH or face.height() < MIN_FACE_HEIGHT:
            continue  # Skip faces that are too small

        landmarks = predictor(gray, face)

        # Get the left and right eye landmarks
        left_eye = [(landmarks.part(point).x, landmarks.part(point).y) for point in LEFT_EYE_POINTS]
        right_eye = [(landmarks.part(point).x, landmarks.part(point).y) for point in RIGHT_EYE_POINTS]

        # Calculate the EAR for both eyes
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # Average the EAR for both eyes
        avg_ear = (left_ear + right_ear) / 2.0

        # Check if the EAR is below the threshold indicating closed eyes
        if avg_ear < ear_threshold:
            return True  # Eyes are closed

    return False

def process_images(directory_path,reference_path):
    results = []
    hashes ={}
    yolo_net, output_layers = load_yolo_model()
    
    try:
        criteria = get_criteria(reference_path)
    except FileNotFoundError as e:
        print(e)
        return results
    
    good_files, bad_files, duplicate_files, flagged_files = [], [], [], []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory_path, filename)
            
            try:
                image = load_image(image_path)
                resized_image = resize_image(image)
                if are_eyes_closed(resized_image, ear_threshold=0.2):
                    status = 'Flagged'
                    reason_text = 'closed eye detected'
                    flagged_files.append(filename)
                    results.append({'filename': filename,'status':status,'reason':reason_text})
                    continue
                if detect_duplicates(image_path,hashes,threshold=15):
                    status = 'Duplicate'
                    reason_text = 'Duplicate Image'
                    duplicate_files.append(filename)
                    results.append({'filename': filename,'status':status,'reason':reason_text})
                    continue
                hashes[filename] = hash_image(image_path)

                blur = get_blur(image)
                
                if blur > criteria['blurriness_max']:
                    objects = detect_objects(image,yolo_net,output_layers)
                    if objects:
                        status = 'Good'
                        reason_text = 'Object in focus'
                        good_files.append(filename)
                    else:
                        status = 'Bad'
                        reason_text = 'Object too subtle in a blurr image'
                        bad_files.append(filename)
                    results.append({'filename': filename,'status':status,'reason':reason_text})
                    continue
                
                status, reason_text = evaluate_image_quality(image, criteria)
                #print(f"Evaluation for {filename}: Status = {status}, Reasons = {reason_text}")  # Debugging statement
                
                if status == 'Bad':
                    bad_files.append(filename)
                else:
                    good_files.append(filename)
                
                results.append({'filename': filename, 'status': status, 'reasons': reason_text})
                
                
            except FileNotFoundError as e:
                print(e)
    bash_script = generate_bash_script(good_files,bad_files,duplicate_files,flagged_files)
    print("\nGenerated Bash Script:\n", bash_script)
    return results
           

def evaluate_image_quality(image, criteria):
    
    blurriness, exposure, noise = quality(image)

    reasons = []
    if blurriness > criteria['blurriness_max']:
        reasons.append('Too much blurriness')
    if exposure < criteria['exposure_min']:
        reasons.append('Exposure too low')
    if exposure > criteria['exposure_max']:
        reasons.append('Exposure too high')
    if noise > criteria['noise_max']:
        reasons.append('Noise too high')

    status = 'Bad' if reasons else 'Good'
    reason_text = ', '.join(reasons) if reasons else 'None'
    return status, reason_text
    
def generate_bash_script(good_files, bad_files, duplicate_files,flagged_files):
    """Generate the bash script for organizing the files."""
    bash_script = "mkdir \"Good\"\nmkdir \"Bad\"\nmkdir \"Duplicate\"\nmkdir \"Flagged\"\n"
    
    for filename in good_files:
        bash_script += f"move \"{filename}\" Good/\n"
    
    for filename in bad_files:
        bash_script += f"move \"{filename}\" Bad/\n"
        
    for filename in duplicate_files:
        bash_script += f"move \"{filename}\" Duplicate/\n"
            
    for filename in flagged_files:
        bash_script += f"move \"{filename}\" Flagged/\n"
    
    return bash_script   
    
def main():
    directory_path = 'uploads'
    reference_path = 'reference/reference.jpg'
    
    time_start = time.time()

    results = process_images(directory_path, reference_path)
    time_end = time.time()
    runtime = time_end - time_start
    print(runtime)
    print("\nResults:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
            
    
    
    