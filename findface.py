from tkinter.tix import IMAGE
import cv2
import mediapipe as mp
from pathlib import Path
import shutil

mp_face_detection = mp.solutions.face_detection



ROOT_DIR = "flickr-10k"
# For static images:
# IMAGE_FILES = ["images/office/50672688_f25a1cfde8_b.jpg","images/office/25483791_f7c0cab1ae_o.jpg"]
IMAGE_FILES = list(map(lambda p:p.as_posix(), Path(ROOT_DIR).glob("*.jpg")))



face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)

def should_reserve(file):
    try:
        image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return 0.
        else:
            return sum([person.score[0] for person in results.detections])
    except:
        return 1. # skip



def should_reserve_batch(filelist, min_detection_confidence=0.25):
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_detection_confidence)
    reserve_list = []
    for file in filelist:
        try:
            image = cv2.imread(file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            reserve_list.append(sum([person.score[0] for person in results.detections]))
        except:
            reserve_list.append(1.)
    return reserve_list



import tqdm
import pprint
from multiprocessing import Pool, cpu_count
from itertools import chain
import json


batchsize = 20
_len = len(IMAGE_FILES)
start_end_indexs = zip(range(0, _len-batchsize, batchsize), range(batchsize, _len, batchsize))
start_end_indexs = chain(start_end_indexs, [ (_len//batchsize * batchsize, _len)])


filelist_list = [IMAGE_FILES[start:end] for start, end in start_end_indexs]
with Pool(cpu_count()//2) as pool:
    masks = pool.map(should_reserve_batch, filelist_list)
    mask = chain(*masks)

# mask = []
# for start, end in tqdm.tqdm(start_end_indexs, total = _len//batchsize):
#     minconf = list(map(should_reserve, IMAGE_FILES[start:end]))
#     mask.extend(minconf)

print(
    json.dumps([ {"path":imgfile, "reserve":reserve} for imgfile, reserve in zip(IMAGE_FILES, mask)], ensure_ascii=False),
    file=open("contains_human.txt", "w")
)
