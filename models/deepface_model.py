from deepface import DeepFace
import tempfile
import cv2
import os



class DeepFaceModel:
    def __init__(self, model_name="Facenet512", detector_backend="retinaface",distance_metric="euclidean"):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric

    def _save_temp(self, img):
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(path, img[:, :, ::-1])  
        return path    

    def verify(self, face1, face2):
        p1 = self._save_temp(face1)

        result = DeepFace.verify(p1, face2, model_name=self.model_name)
        os.remove(p1)

        return result
    
    def recognition(self, face, folder_path):
        p1 = self._save_temp(face)

        recognition = DeepFace.find(p1, db_path = folder_path, model_name = self.model_name, detector_backend="retinaface",distance_metric="euclidean")
        return recognition
    
    def get_best_match(self, recognition_result):
        df = recognition_result[0]     
        best = df.iloc[0]             
        return best["identity"], best["distance"]
    

    def filter_unique_best_matches(self, recognition_result):
        df = recognition_result[0]
        df = df.copy()
        df["person"] = df["identity"].apply(lambda p: os.path.basename(os.path.dirname(p)))
        best_df = df.loc[df.groupby("person")["distance"].idxmin()]
        best_df = best_df.sort_values("distance")
        return best_df.reset_index(drop=True)
    

    def extract_faces(self, face):
        return DeepFace.extract_faces(
            img_path=face,
            detector_backend='retinaface',
            enforce_detection=False
        )