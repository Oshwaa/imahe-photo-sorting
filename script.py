import cv2
import os
import imagehash
from PIL import Image
import time

def load_image(image_path):
    """Load an image from the specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def detect_blurriness(image):
    """Detect blurriness using the variance of the Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def detect_exposure(image):
    """Detect exposure by calculating the average pixel value."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_pixel_value = gray.mean()
    return average_pixel_value

def detect_noise(image):
    """Detect noise by calculating the standard deviation of pixel values."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stddev = gray.std()
    return stddev

def hash_image(image_path):
    """Generate a perceptual hash for the image."""
    image = Image.open(image_path)
    hash_value = imagehash.phash(image)
    return hash_value

def get_reference_criteria(reference_image):
    """Get criteria from the reference image to set thresholds."""
    blurriness = detect_blurriness(reference_image)
    exposure = detect_exposure(reference_image)
    noise = detect_noise(reference_image)

    # Define thresholds based on reference image criteria with a range
    return {
        'blurriness_max': blurriness * 1.4,  # Allow 40% higher blurriness
        'exposure_min': exposure * 0.7,  # Allow 30% lower exposure
        'exposure_max': exposure * 1.4,  # Allow 40% higher exposure
        'noise_max': noise * 1.9  # Allow 90% higher noise 
    }

def process_images(directory_path, reference_path):
    """Process all images in the specified directory and detect duplicates."""
    results = []
    hashes = {}
    
    reference_image = load_image(reference_path)
    criteria = get_reference_criteria(reference_image)
    
    print("Reference Criteria:")
    print(criteria)
    
    good_files = []
    bad_files = []
    duplicate_files = []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory_path, filename)
            try:
                image = load_image(image_path)
                
                # Check for duplicates using perceptual hash
                image_hash = hash_image(image_path)
                if any(image_hash - existing_hash < 5 for existing_hash in hashes):  # Tolerance of 5 for slight differences
                    duplicate_files.append(filename)
                    results.append({
                        'filename': filename,
                        'status': 'Duplicate',
                        'reasons': 'Similar to an existing image'
                    })
                    continue
                
                hashes[image_hash] = filename
                
                # Check criteria
                blurriness = detect_blurriness(image)
                exposure = detect_exposure(image)
                noise = detect_noise(image)
                
                
                print(f"\nProcessing: {filename}")
                print(f"Blurriness: {blurriness}, Criteria: <= {criteria['blurriness_max']}")
                print(f"Exposure: {exposure}, Criteria: {criteria['exposure_min']} - {criteria['exposure_max']}")
                print(f"Noise: {noise}, Criteria: <= {criteria['noise_max']}")
                
                
                reasons = []
                if blurriness > criteria['blurriness_max']:
                    reasons.append('Too much blurriness')
                if exposure < criteria['exposure_min']:
                    reasons.append('Exposure too low')
                if exposure > criteria['exposure_max']:
                    reasons.append('Exposure too high')
                if noise > criteria['noise_max']:
                    reasons.append('Noise too high')
                
                if reasons:
                    status = 'Bad'
                    reason_text = ', '.join(reasons)
                    bad_files.append(filename)
                else:
                    status = 'Good'
                    reason_text = 'None'
                    good_files.append(filename)
                
                results.append({
                    'filename': filename,
                    'status': status,
                    'reasons': reason_text
                })
                
                
                print(f"Status: {status}")
                if status == 'Bad':
                    print(f"Reasons: {reason_text}")
                
            except FileNotFoundError as e:
                print(e)
    
    # Generate the bash script content
    bash_script = ""
    bash_script += "mkdir \"Good\"\n"
    bash_script += "mkdir \"Bad\"\n"
    bash_script += "mkdir \"Duplicate\"\n\n"

    for filename in good_files:
        bash_script += f"move \"{filename}\" Good/\n"

    for filename in bad_files:
        bash_script += f"move \"{filename}\" Bad/\n"
    
    for filename in duplicate_files:
        bash_script += f"move \"{filename}\" Duplicate/\n"
    
    print("\nGenerated Bash Script:")
    print(bash_script)
    
    return results

def main():
    directory_path = 'uploads'  
    reference_path = 'reference/reference.jpg'  
    start_time = time.time()
    results = process_images(directory_path, reference_path)
    end_time = time.time()
    print("\nImage Results:")
    for result in results:
        print(f"File: {result['filename']}, Status: {result['status']}, Reasons: {result['reasons']}")
        
    time_taken = end_time - start_time
    print(f"\nTime taken to process images: {time_taken:.2f} seconds")

if __name__ == "__main__":
    main()
