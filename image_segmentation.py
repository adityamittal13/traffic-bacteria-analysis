import numpy as np
import cv2

def count_cells(filename):
    img = cv2.imread(filename)
    image_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_array, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    areas = [cv2.contourArea(cnt) for cnt in cnts if 20 < cv2.contourArea(cnt)]

    quartile_cutoff = np.percentile(areas, 85)
    low_areas = [area for area in areas if area < quartile_cutoff]
    high_areas = [area for area in areas if area >= quartile_cutoff]

    count = len(low_areas)
    for high_area in high_areas:
        count += round(high_area/quartile_cutoff)

    sorted_cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
    print(list(map(lambda x: cv2.contourArea(x), sorted_cnts)))

    for idx, cnt in enumerate(sorted_cnts[-5:-1]):
        x, y = [], []
        x, y, w, h = cv2.boundingRect(cnt)
        ROI = img[y: y+h, x: x+w]
        cv2.imwrite(f'{filename}_{idx}.png', ROI)
     
    return count

print(count_cells('./experiment-data/Doherty Inside.jpeg'))