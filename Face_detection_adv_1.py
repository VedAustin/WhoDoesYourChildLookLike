# To use a small model to auto put pics in the right folder

# Load model

from sklearn.externals import joblib
import cv2
from os import listdir
from os.path import join
from save_img import rescalePx, aggTransform, reshapePx, resizePx

face_cascade = cv2.CascadeClassifier('D:\\Documents\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
classifier = joblib.load('face_detect_model_1.pkl')

# Read an image
pic_path = 'D:\\Pictures\\Dad_Google_photos\\'
#pic_path = 'D:\\Pictures\\Mom_Google_photos\\'
#pic_path = 'D:\\Pictures\\Child_Google_photos\\'
files = listdir(pic_path)

for afile in files:
	path2image = join(pic_path,afile)
	afile = afile[:-4]
	print afile
	gray_img = cv2.imread(path2image,1)
	# Detect Face
	faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

	if len(faces)>0:
		n = len(faces)
		for (x,y,w,h) in faces:
			cropped_img = gray_img[y:y+h,x:x+w]
			# For each face: detect category
			# Convert the image to numbers
			images_array = aggTransform(cropped_img)
			predicted_label = classifier.predict(images_array)
			# save the face in appropriate folder
			if predicted_label == 0.:
				cropped_path = 'D:\\Pictures\\Dad_cropped\\'
			else:
				cropped_path = 'D:\\Pictures\\Mom_cropped\\'				
			crop_path_full = join(cropped_path,afile + '_' + str(n) + '.png') 
			#print crop_path_full
			cv2.imwrite(crop_path_full,cropped_img)
			n -= 1
		else:
			pass