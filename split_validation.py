import os
species = os.listdir('dataset/images/lab/train') #gets the name of all directories
os.makedirs('dataset/images/lab/val') #Creates the val directory
for specie in species:
	if not os.path.exists('dataset/images/lab/val/'+specie):
		os.makedirs('dataset/images/lab/val/'+specie) #Creates the val directory
	images = os.listdir('dataset/images/lab/train/'+specie) #gets the images names
	val_size = int(round(len(images)*0.2)) #gets the number of val images (20%)

	for i in range(val_size): #change the folder of val images
		os.rename('dataset/images/lab/train/'+specie+'/'+images[i], 'dataset/images/lab/val/'+specie+'/'+images[i])
