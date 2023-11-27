import numpy as np
import cv2

def count_cells(filename):
    img = cv2.imread(filename)
    image_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(image_array, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    kernel = np.ones((5,5), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.erode(image_array, kernel, iterations=1)
    opening = cv2.dilate(opening, kernel, iterations=1)
    
    # Find background region by dialating
    # sure_bg = cv2.dilate(opening, kernel, iterations=1)

    # Find foreground region
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3) 

    cv2.imshow("input", opening)
    cv2.waitKey(0)

    ret2, sure_fg = cv2.threshold(opening, 0.1 * opening.max(), 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
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
        cv2.imwrite(f'{filename.replace(".png", "")}_{idx}.png', ROI)
     
    return count

print(count_cells('./experiment-data/Wean Outside Circle.png'))