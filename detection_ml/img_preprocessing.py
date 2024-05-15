import cv2 as cv
import os

folder_path = "/data_img_Total"
images_traitees = []

for filename in os.listdir(folder_path):
  image_path = os.path.join(folder_path, filename)
  image = cv.imread(image_path)
  cropped_img = image[800:1800, 1800:2800]
  gray_image = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
  images_traitees.append(gray_image)
 