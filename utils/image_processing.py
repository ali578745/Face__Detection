import matplotlib.pyplot as plt
import cv2
import os

def get_rgb_image(image_path):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def show_verification_result(result_dict):
    print("Loading Results...")
    print("\nDEEPFACE RESULT:\n")
    for key, value in result_dict.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")




def show_verification_visual(test_image, reference_image_path, verify_result):
    ref_bgr = cv2.imread(reference_image_path)
    ref_rgb = ref_bgr[:, :, ::-1] 

    verified = verify_result.get("verified", False)
    distance = verify_result.get("distance", 0.0)
    threshold = verify_result.get("threshold", 25.0) 

    max_dist = 60.0
    confidence = max(0.0, (max_dist - distance) / max_dist) * 100

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(test_image)
    ax[0].set_title("Query Face")
    ax[0].axis("off")
    
    ax[1].imshow(ref_rgb)
    status = "✅ MATCH" if verified else "❌ NO MATCH"
    ax[1].set_title(
        f"Reference Face\n{status}\n"
        f"Distance: {distance:.2f} (Threshold: {threshold:.2f})\n"
        f"Confidence: {confidence:.1f}%",
        fontsize=11
    )
    ax[1].axis("off")
    
    plt.tight_layout()
    plt.show()            



def show_recognition_result(matched_face_path, distance):
     person_name = os.path.basename(os.path.dirname(matched_face_path))
     print("\nRECOGNITION RESULT:")
     print(f"Matched Person: {person_name}")
     print(f"Matched Image : {matched_face_path}")
     print(f"Distance      : {distance:.4f}")

     max_dist = 60.0
     confidence = max(0.0, (max_dist - distance) / max_dist) * 100
     print(f"Confidence    : {confidence:.1f}%")



def show_recognition_visual(test_image, matched_face_path, distance):
    person_name = os.path.basename(os.path.dirname(matched_face_path))
    left = test_image
    bgr = cv2.imread(matched_face_path)
    right = bgr[:, :, ::-1]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(left)
    ax[0].set_title("Target Face")
    ax[0].axis("off")

    ax[1].imshow(right)
    ax[1].set_title(f" Detected Person:{person_name}\nDistance: {distance:.3f}", fontsize=12)
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def capture_from_webcam():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print('Cannot open webcam')
        return None

    print('Webcam active. Press SPACE to capture, ESC to cancel.')
    while True:
        ret, bgr_frame = capture.read()
        if not ret:
            print('Failed to grab frame')
            break
        cv2.imshow('Webcam - Press SPACE to capture, ESC to exit', bgr_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: 
            print('Capture canceled.')
            break
        elif key == 32:
            print("Image captured!")
            capture.release()
            cv2.destroyAllWindows()
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            return rgb_frame

    capture.release()
    cv2.destroyAllWindows()
    return None