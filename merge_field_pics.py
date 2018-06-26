import os
species = os.listdir('dataset/images/field') #gets the name of all directories in field images
for specie in species:
	images = os.listdir('dataset/images/field/'+specie) #gets the images names
	test_size = int(round(len(images)*0.2)) #gets the number of test images (20%)
 
	for i in range(test_size): #change the folder of test images
		os.rename('dataset/images/field/'+specie+'/'+images[i], 'dataset/images/lab/test/'+specie+'/'+images[i])
		
	images = os.listdir('dataset/images/field/'+specie) #get the remaining images to train
	for image in images:
		os.rename('dataset/images/field/'+specie+'/'+image, 'dataset/images/lab/train/'+specie+'/'+image)
