import os
import tqdm as tq
import numpy as np
import cv2
import easyocr
import time

reader = easyocr.Reader(['en'], gpu = False) # only english for now

output_path = '/home/md.hassan/charts/ChartIE/results_det/'
image_path = '/home/md.hassan/charts/s_CornerNet/synth_data/data/line/figqa/val/images/'
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1

for f in os.listdir(output_path):
    os.remove(output_path+'/'+f) 

ctr = 0
time_ = 0
for image_name in tq.tqdm(os.listdir(image_path)[:20]):
    image = cv2.imread(image_path + image_name)
    t = time.time()
    results = reader.readtext(image, rotation_info = [270]) # assuming 270 deg rotation for vertical text
    time_ += time.time() - t
    for result in results:
        r = np.array(result[0]).astype(int)
        cv2.rectangle(image, r[0], r[2], (0, 255, 0), 1)  
        cv2.putText(image, result[1], r[0], font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)          
    cv2.imwrite(output_path + str(ctr) + '.png', image)
    ctr += 1
print("Avg OCR time = {}".format(time_/ctr))
# 0.3744 sec with GPU
# 1.7678 sec without GPU
