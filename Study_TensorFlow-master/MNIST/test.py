import cv2
img = cv2.imread("test.png")
start_y= 0
end_y = 0

for slice in range(3):
    start_y = start_y + 27
    end_y = end_y + 28

    # crop_img = img[10+((slice)*50) : 10+(slice*50), 10+((slice)*50) : 10+(slice*50)] # Crop from x, y, w, h -> 100, 200, 300, 400 [시작높이: 끝높이, 시작길이: 끝길이 ]
    # crop_img = img[start_y : end_y, 0 : 10+(slice*10)] # Crop from x, y, w, h -> 100, 200, 300, 400 [시작높이: 끝높이, 시작길이: 끝길이 ]
    crop_img = img[0:100, 0:100] # Crop from x, y, w, h -> 100, 200, 300, 400 [시작높이: 끝높이, 시작길이: 끝길이 ]
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)

a = crop_img
cv2.imwrite('test_crop.png',a)

# https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python