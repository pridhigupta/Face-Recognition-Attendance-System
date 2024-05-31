import cv2
import numpy as npy
import face_recognition as face_rec

def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

prerna=face_rec.load_image_file("photos/prerna.jpg")
prerna=cv2.cvtColor(prerna,cv2.COLOR_BGR2RGB)
prerna=resize(prerna,0.90)
prerna_test=face_rec.load_image_file("photos/pridhi.jpg")
prerna_test=cv2.cvtColor(prerna_test,cv2.COLOR_BGR2RGB)
prerna_test=resize(prerna_test,0.90)

faceLocation_prerna= face_rec.face_locations(prerna)[0]
encode_prerna = face_rec.face_encodings(prerna)[0]
cv2.rectangle(prerna, (faceLocation_prerna[3], faceLocation_prerna[0]), (faceLocation_prerna[1], faceLocation_prerna[2]), (255, 0, 255), 3)

faceLocation_prernatest= face_rec.face_locations(prerna_test)[0]
encode_prernatest = face_rec.face_encodings(prerna_test)[0]
cv2.rectangle(prerna_test, (faceLocation_prernatest[3], faceLocation_prernatest[0]), (faceLocation_prernatest[1], faceLocation_prernatest[2]), (255, 0, 255), 3)


results = face_rec.compare_faces([encode_prerna], encode_prernatest)
print(results)
cv2.putText(prerna_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow("main_img",prerna)
cv2.imshow("test_img",prerna_test)

cv2.waitKey(0)
cv2.destroyAllWindows()