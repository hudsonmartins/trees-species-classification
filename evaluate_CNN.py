import glob, os
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

test_dir = 'dataset/images/lab/test'
model = load_model('model_field2.h5')

def to_one_hot(X):
	'''
		Converts the input to one-hot encoding
	'''
	one_hot = np.zeros((len(X), 185))
	row = 0
	for value in X:
		one_hot[int(row), int(value)] = 1
		row += 1
	return one_hot


species = sorted(os.listdir(test_dir))	#Get species names sorted
predictions = []
target = []

for specie in species:
	for image in glob.glob(test_dir+"/"+specie+'/*.jpg'):
		#Opens image as vector 
		img = Image.open(image)
		img = np.asarray(img)
		img = np.expand_dims(img, axis=0)
	
		pred = model.predict(img) #Makes the prediction
		predictions.append(np.argmax(pred))
		target.append(species.index(specie))
	
target = to_one_hot(target) #Convert to one-hot-encoding
predictions = to_one_hot(predictions)

print classification_report(target, predictions)
print "F1: ", f1_score(target, predictions, average='macro')
print "Acc: ", accuracy_score(target, predictions)
