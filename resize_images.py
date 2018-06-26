from PIL import Image
import glob, os.path

classes = os.listdir('dataset/images/lab/train') #Get the name of the classes
	
def resize_from_path(input_path): 
  for specie in classes:
    path = input_path+'/'+specie #gets the path of the image
    
    for image in glob.glob(path+'/*.jpg'): 
      img = Image.open(image)     #Open the image
      #Crops the image in a square before resizing		
      if(img.size[0] < img.size[1]):
        center = [img.size[0]/2, img.size[1]/2]   #Gets the center of the image
        img = img.crop((center[0]-(img.size[0]/2), center[1]-(img.size[0]/2), center[0]+(img.size[0]/2), center[1]+(img.size[0]/2)))
      else:
        center = [img.size[0]/2, img.size[1]/2]   #Gets the center of the image
        img = img.crop((center[0]-(img.size[1]/2), center[1]-(img.size[1]/2), center[0]+(img.size[1]/2), center[1]+(img.size[1]/2)))
        
      #Resizing
      img = img.resize((300, 300))
      img.save(image)
      
# ------------- Resizing all the images -----------
resize_from_path('dataset/images/lab/train')
resize_from_path('dataset/images/lab/test')
resize_from_path('dataset/images/lab/val')
