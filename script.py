import cv2
import os
import imagehash
from PIL import Image
import numpy as np
import time

def load_image(image_path):
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

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

    return objects

def crop_object(image, x, y, w, h):
    
    return image[y:y+h, x:x+w]

def detect_blurriness(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def detect_exposure(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.mean()

def detect_noise(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()

def hash_image(image_path):
   
    image = Image.open(image_path)
    hash_value = imagehash.phash(image)
    print(f"Generated Hash for {image_path}: {hash_value}")  # Debug
    return hash_value

def get_reference_criteria(reference_image):
    
    blurriness = detect_blurriness(reference_image)
    exposure = detect_exposure(reference_image)
    noise = detect_noise(reference_image)

    criteria = {
        'blurriness_max': blurriness * 4,  # FINE TUNE
        'exposure_min': exposure * 0.7,  
        'exposure_max': exposure * 4,  
        'noise_max': noise * 2.4  
    }

    print(f"Reference Criteria: {criteria}")  # Debug
    return criteria

def load_yolo_model():
    try:
        yolo_net = cv2.dnn.readNet('yolo/yolov3.weights', 'yolo/yolov3.cfg')
        layer_names = yolo_net.getLayerNames()
        unconnected_out_layers = yolo_net.getUnconnectedOutLayers()
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
        return yolo_net, output_layers
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        raise

def detect_duplicates(image_path, hashes, threshold=5):
    image_hash = hash_image(image_path)
    print(f"Checking duplicates for {image_path} with hash {image_hash}")  # Debug
    is_duplicate = any(
        isinstance(existing_hash, imagehash.ImageHash) and
        (image_hash - existing_hash) < threshold
        for existing_hash in hashes.values()
    )
    print(f"Is {image_path} a duplicate? {is_duplicate}")  # Debug
    return is_duplicate

def process_images(directory_path, reference_path):
    
    results = []
    hashes = {}
    yolo_net, output_layers = load_yolo_model()

    # Load reference image and get criteria
    try:
        reference_image = load_image(reference_path)
        criteria = get_reference_criteria(reference_image)
    except FileNotFoundError as e:
        print(e)
        return results

    good_files, bad_files, duplicate_files = [], [], []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory_path, filename)
            try:
                image = load_image(image_path)

                # Check for duplicates first
                if detect_duplicates(image_path, hashes, threshold=5):
                    status = 'Duplicate'
                    reason_text = 'Duplicate image'
                    duplicate_files.append(filename)
                    results.append({'filename': filename, 'status': status, 'reasons': reason_text})
                    continue  # Skip further processing for this image
                
                # Store the hash of the image
                hashes[filename] = hash_image(image_path)
                
                blurriness = detect_blurriness(image)
                print(f"Blurriness for {filename}: {blurriness}")  # Debug
                
                if blurriness > criteria['blurriness_max']:
                    # If image is too blurry, only check for objects
                    objects = detect_objects(image, yolo_net, output_layers)
                    if objects:
                        status = 'Good'
                        reason_text = 'Objects detected in a blurry image'
                        good_files.append(filename)
                    else:
                        status = 'Bad'
                        reason_text = 'No objects detected in a blurry image'
                        bad_files.append(filename)
                    
                    results.append({'filename': filename, 'status': status, 'reasons': reason_text})
                    continue  # Skip further processing for this image

                # For non-blurry images, check all criteria
                status, reason_text = evaluate_image_quality(image, criteria)
                print(f"Evaluation for {filename}: Status = {status}, Reasons = {reason_text}")  # Debugging statement
                
                if status == 'Bad':
                    bad_files.append(filename)
                else:
                    good_files.append(filename)
                
                results.append({'filename': filename, 'status': status, 'reasons': reason_text})

            except FileNotFoundError as e:
                print(e)

    bash_script = generate_bash_script(good_files, bad_files, duplicate_files)
    print("\nGenerated Bash Script:\n", bash_script)
    return results

def evaluate_image_quality(image, criteria):
    """Evaluate image quality based on criteria and return status and reasons."""
    blurriness = detect_blurriness(image)
    exposure = detect_exposure(image)
    noise = detect_noise(image)

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

def generate_bash_script(good_files, bad_files, duplicate_files):
    """Generate the bash script for organizing the files."""
    bash_script = "mkdir \"Good\"\nmkdir \"Bad\"\nmkdir \"Duplicate\"\n"
    
    for filename in good_files:
        bash_script += f"move \"{filename}\" Good/\n"
    
    for filename in bad_files:
        bash_script += f"move \"{filename}\" Bad/\n"
    
    for filename in duplicate_files:
        bash_script += f"move \"{filename}\" Duplicate/\n"
    
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
