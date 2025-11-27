import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from models.deepface_model import DeepFaceModel
from utils.image_processing import (
    get_rgb_image,
    capture_from_webcam,
    show_verification_result,
    show_verification_visual,
    show_recognition_result,
    show_recognition_visual
    
)
import argparse
import traceback

def main():
    parser = argparse.ArgumentParser(description='Face Recognition with DeepFace')
    parser.add_argument('--test', type=str, default=None, help='Input Image to Test Model')
    parser.add_argument('--webcam', action='store_true', help='captures frames from webcam')
    parser.add_argument('--random', type=str, required=None, help='Random Image to compare with Input image')
    parser.add_argument('--database', type=str, default=None, help='Path folder for Recognition')

    args = parser.parse_args()
    if args.test is None and args.webcam is None:
        parser.error('Provide --test image or --webcam image')
    if args.test and args.webcam:
        parser.error('Use --test or --webcam : not both')

    if args.random is None and args.database is None:
        parser.error('You must provide either --random (for verification) or --database (for recognition).')
    if args.random and args.database:
        parser.error('Use --random or --database : not both')

    rgb_testimage = None
    if args.test:    
     try:
        rgb_testimage = get_rgb_image(args.test)
        print("Test image loaded successfully")
     except Exception as e:
        print(f"Failed to load test image: {e}")
        return
    elif args.webcam:
        print('Opening webcam... Press SPACE to capture, ESC to cancel.')
        rgb_testimage = capture_from_webcam()
        if rgb_testimage is None:
            print('No image captured from webcam.')
            return 
    model = DeepFaceModel() 

    #verification -- comparing if images are similar
    if args.random is not None:  
        try:
            print('Running Verification')
            result = model.verify(rgb_testimage, args.random)
        except Exception as e:
            print('verification failed')
            traceback.print_exc()
            return
        show_verification_result(result)
        show_verification_visual(rgb_testimage, args.random, result)
        print("***"*56)

    #Recognition -- comparing input image from a database of images
    if args.database is not None:
        try:
            result2 = model.recognition(rgb_testimage, args.database)
            matched_face_path, distance = model.get_best_match(result2)
            if matched_face_path:
                show_recognition_result(matched_face_path, distance)
                show_recognition_visual(rgb_testimage, matched_face_path, distance)
            else:
                print("No match found in database.")
        except Exception as e:
            print("Recognition failed")
            traceback.print_exc()

if __name__ == "__main__":
    main()