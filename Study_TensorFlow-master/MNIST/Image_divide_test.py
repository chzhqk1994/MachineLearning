import cv2
img = cv2.imread("test.png")
crop_img = img[0:, 100:] # Crop from x, y, w, h -> 100, 200, 300, 400 [시작높이: 끝높이, 시작길이: 끝길이 ]
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

a = crop_img
cv2.imwrite('crop_img.png',a)
