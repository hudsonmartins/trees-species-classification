import glob, os
import numpy as np
from PIL import Image
from keras.models import load_model, Sequential, Model 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

def extract_features(path):
	
	model = load_model('model_field.h5') #Loads the CNN
	feature_extractor = Sequential()
	
	for layer in model.layers[:-4]: #Removing the FC layers
		feature_extractor.add(layer)

	species = sorted(os.listdir(path)) #Get the species names
	features = []
	target = []
	print "Extracting features"
	for specie in species:
		print specie
		for image in glob.glob(path+"/"+specie+'/*.jpg'): 
			#Open the image as array
			img = Image.open(image)
			img = np.asarray(img)
			img = np.expand_dims(img, axis=0)
			
			#Inputs the image on the CNN
			output = feature_extractor.predict(img)
			output=output.reshape(output.shape[1]) #reshaping the output to a vector (640 features)
			
			features.append(output)
			target.append(species.index(specie))
	return features, target

def train_rfc():
	print "Training Random Forest"
	rfc = RandomForestClassifier(n_estimators = 200, verbose=True)  #Initialize the Random Forest Classifier
	X, y = extract_features('dataset/images/lab/train')             #Gets the features from training set
	rfc.fit(X, y) 							#Training the RFC
	return rfc
	
def predict(rfc):
	X, y_targ = extract_features('dataset/images/lab/test') 	#Get features from test set
	print "Predicting test set"
	y_pred = rfc.predict(X)						#Makes a prediction
	print classification_report(y_targ, y_pred)
	print "F1: ", f1_score(y_targ, y_pred, average='macro')
	print "Acc: ", accuracy_score(y_targ, y_pred)
	cf = confusion_matrix(y_targ, y_pred)				#Build Confusion Matrix
	np.savetxt("results/confusion_matrix_rf", cf, fmt='%i')
	
rfc = train_rfc()
predict(rfc)
