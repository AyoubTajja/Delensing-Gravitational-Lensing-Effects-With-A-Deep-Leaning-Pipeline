from import_and_installations import *
from Dataset_generation import n,dataset,labelset

def shapelet_reconstruction(n,choice_type_shapelet,dataset,beta,nmax,x,y,deltaPix):
  
  
  dataset_reconstructed=[]
  coeff_dataset=[]

  ## Choose the type of shapelets : Polar or Cartesian shapelets
  shapeletSet = choice_type_shapelet
  for ind in tqdm(range(n)):
    coeff_dataset_local=[]
    dataset_reconstructed_local=[]
    
    image_1d = util.image2array(dataset[ind][0])
    coeff_ngc = shapeletSet.decomposition(image_1d, x, y, nmax, beta, deltaPix=1, center_x=0, center_y=0) 
    # reconstruct NGC1300 with the shapelet coefficients
    image_reconstructed = shapeletSet.function(x, y, coeff_ngc, nmax, beta, center_x=0, center_y=0)
    # turn 1d array back into 2d image
    image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image
    dataset_reconstructed_local.append(image_reconstructed_2d)
    coeff_dataset_local.append(coeff_ngc)

  
    image_1d = util.image2array(dataset[ind][1])  
    # decompose image and return the shapelet coefficients
    coeff_ngc = shapeletSet.decomposition(image_1d, x, y, nmax, beta, deltaPix=1, center_x=0, center_y=0) 
    # reconstruct NGC1300 with the shapelet coefficients
    image_reconstructed = shapeletSet.function(x, y, coeff_ngc, nmax, beta, center_x=0, center_y=0)
    # turn 1d array back into 2d image
    image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image
    dataset_reconstructed_local.append(image_reconstructed_2d)
    coeff_dataset_local.append(coeff_ngc)


    image_1d = util.image2array(dataset[ind][2])  
    # decompose image and return the shapelet coefficients
    coeff_ngc = shapeletSet.decomposition(image_1d, x, y, nmax, beta, deltaPix=1, center_x=0, center_y=0) 
    # reconstruct NGC1300 with the shapelet coefficients
    image_reconstructed = shapeletSet.function(x, y, coeff_ngc, nmax, beta, center_x=0, center_y=0)
    # turn 1d array back into 2d image
    image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image
    dataset_reconstructed_local.append(image_reconstructed_2d)
    coeff_dataset_local.append(coeff_ngc)

    dataset_reconstructed.append(np.array(dataset_reconstructed_local))
    coeff_dataset.append(np.array(coeff_dataset_local))
  
  dataset_reconstructed=np.array(dataset_reconstructed)
  coeff_dataset=np.array(coeff_dataset)

  return(dataset_reconstructed,coeff_dataset)


n_=n
choice_type_shapelet=ShapeletSet()
dataset_=dataset

numPix=128
nmax=170

theta_max=numPix
theta_min=theta_max/(nmax+1)

beta=np.sqrt(theta_max*theta_min)

deltaPix=theta_min
print('\nbeta = '+str(beta))
print('nmax = '+str(nmax))

x, y = util.make_grid(numPix, deltaPix)

dataset_reconstructed,coeff_dataset=shapelet_reconstruction(n_,choice_type_shapelet,dataset_,beta,nmax,x,y,deltaPix)