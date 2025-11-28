import os
import cv2
from models.deepface_model import DeepFaceModel

def run_inference(args):
    if not args.database:
        raise SystemExit("Provide --database for recognition")

    model = DeepFaceModel()

    is_image = False
    if os.path.isfile(args.source) and args.source.lower().endswith(('.jpg', '.jpeg', '.png')):
        is_image = True
        bgr_frame = cv2.imread(args.source)
        if bgr_frame is None:
            raise SystemExit(f'Cannot load image: {args.source}')
        frames = [bgr_frame]
    else:
        source = 0 if args.source == 'webcam' else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f'Cannot open source: {args.source}')
        frames = cap

    print('Running face recognition. Press \'q\' to exit.')

    while True:
        if is_image:
            bgr_frame = frames[0]
            ret = True
        else:
            ret, bgr_frame = frames.read()
            if not ret:
                break
            max_width = 800
            height, width = bgr_frame.shape[:2]
            if width > max_width:
             scale = max_width / width
             new_width = max_width
             new_height = int(height * scale)
             bgr_frame = cv2.resize(bgr_frame, (new_width, new_height))

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        try:
            face_objs = model.extract_faces(rgb_frame)
        except Exception:
            face_objs = []

        for face_obj in face_objs:
            area = face_obj['facial_area']
            x, y, w, h = area['x'], area['y'], area['w'], area['h']

        
            cropped_rgb = rgb_frame[y:y+h, x:x+w]
            if cropped_rgb.size == 0:
                continue

        
            label = 'Unknown'
            try:
                results = model.recognition(cropped_rgb, args.database)
                if results and len(results) > 0:
                    #identity, distance = model.get_best_match(results)
                    #threshold = 23.56
                    #if identity and distance <= threshold:  
                        #label = os.path.basename(os.path.dirname(identity))
                    filtered = model.filter_unique_best_matches(results)
                    if len(filtered)>0:
                        row = filtered.iloc[0]
                        identity = row["identity"]
                        distance = row["distance"]
                        threshold = 23.56
                        if distance <= threshold:
                            label = os.path.basename(os.path.dirname(identity))    
            except Exception:
                pass

            
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(bgr_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('DeepFace Recognition', bgr_frame)
        key = cv2.waitKey(1 if not is_image else 0) & 0xFF
        if key == ord('q') or is_image:
            break

    if not is_image:
        frames.release()
    cv2.destroyAllWindows()