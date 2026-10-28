import cv2
image=cv2.imread('images/test.jpg')

if image is None:
    raise FileNotFoundError("File Not Found ")
print('Image Shape: ',image.shape)
print('Image Data type', image.dtype)
print('Image dimension',image.ndim)


resized=cv2.resize(image,(640,480))
print('Original', image.shape)
print('Resized: ', resized.shape)

cv2.imshow('Origianal Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

success=cv2.imwrite('outputs/resized.jpg',resized)
if not success:
    raise OSError("Image not saved")