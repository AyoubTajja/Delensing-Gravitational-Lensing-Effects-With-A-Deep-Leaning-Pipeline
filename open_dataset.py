from import_and_installations import *

def open_dataset(source):
  dataset=[]
    
  for filename in tqdm(os.listdir(source)):
        
        image = matplotlib.image.imread(source + filename)
        image=image[:,:,0:3]
        R=image[:,:,0]
        G=image[:,:,1]
        B=image[:,:,2]
        img= 0.2125* R + 0.7154* G + 0.0721 *B

        dataset.append(np.array(img))
  return (np.array(dataset))