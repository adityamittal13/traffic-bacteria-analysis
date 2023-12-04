import numpy as np
import cv2
import os

def count_cells(filename, external=False):
    orig_folder_name = "external-data" if external else "cropped-imgs"
    folder_name = "ext-seg-imgs" if external else "seg-imgs"
    path_name = "jpg" if external else "jpeg"

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

    # cv2.imshow("input", opening)
    # cv2.waitKey(0)

    ret2, sure_fg = cv2.threshold(opening, 0.1 * opening.max(), 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    sorted_cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
    filtered_sorted_cnts = list(filter(lambda x: cv2.contourArea(x) > 100, sorted_cnts[:-1]))
    print(list(map(lambda x: cv2.contourArea(x), filtered_sorted_cnts)), len(filtered_sorted_cnts))

    for idx, cnt in enumerate(filtered_sorted_cnts):
        x, y = [], []
        x, y, w, h = cv2.boundingRect(cnt)
        ROI = img[y: y+h, x: x+w]
        cv2.imwrite(f'{filename.replace(path_name, "").replace(orig_folder_name, folder_name)}_{idx}.{path_name}', ROI)

for file in os.listdir('./cropped-imgs'):
    print(f"{file}:")
    count_cells(f"./cropped-imgs/{file}")
for file in os.listdir('./external-data'):
    print(f"{file}:")
    count_cells(f"./external-data/{file}", external=True)
