from import_and_installations import *
from Dataset_generation import generating_dataset,x,y,numPix,deltaPix,data_class, psf_class, kwargs_numerics, exp_time, background_rms


def shapelet_reconstruction(n,source_dataset,source_reconstruction,choice_type_shapelet,beta,nmax,x,y,deltaPix):
  variable_dataset=False
    
  if not os.path.exists(source_reconstruction):
    os.makedirs(source_reconstruction)
    os.makedirs(source_reconstruction+'/Lensed_Dataset')
    os.makedirs(source_reconstruction+'/Unlensed_Dataset')
    os.makedirs(source_reconstruction+'/Noisy_Dataset')
    
    os.makedirs(source_reconstruction+'/DataFrame_unlensed_coeff_dataset')
    os.makedirs(source_reconstruction+'/DataFrame_lensed_coeff_dataset')
    os.makedirs(source_reconstruction+'/DataFrame_noisy_coeff_dataset')
    
    variable_dataset=True
    
  if variable_dataset:
      
      unlensed_coeff_dataset=[]
      lensed_coeff_dataset=[]
      noisy_coeff_dataset=[]
     
      ## Choose the type of shapelets : Polar or Cartesian shapelets
      shapeletSet = choice_type_shapelet
      
       
      for i in tqdm(range(n)):
          
        ##Unlensed data reconstruction
        folder_temp=source_dataset+'/Unlensed_Dataset/'    
        image = matplotlib.image.imread(folder_temp + str(i)+'.png')
        image=image[:,:,0:3]
        R=image[:,:,0]
        G=image[:,:,1]
        B=image[:,:,2]
        unlensed_data= 0.2125* R + 0.7154* G + 0.0721 *B
        
        image_1d = util.image2array(unlensed_data)
        coeff_ngc = shapeletSet.decomposition(image_1d, x, y, nmax, beta, deltaPix=deltaPix, center_x=0, center_y=0) 
        # reconstruct the image with the shapelet coefficients
        image_reconstructed = shapeletSet.function(x, y, coeff_ngc, nmax, beta, center_x=0, center_y=0)
        # turn 1d array back into 2d image
        image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image
        image_reconstructed_2d=image_reconstructed_2d/np.max(image_reconstructed_2d)
        # save in the wanted folder
        matplotlib.image.imsave(source_reconstruction+'/Unlensed_Dataset/'+str(i)+'.png',image_reconstructed_2d,vmin=np.min(image_reconstructed_2d),vmax=np.max(image_reconstructed_2d))
        unlensed_coeff_dataset.append(np.array(coeff_ngc))
        
        ##Lensed data reconstruction
        folder_temp=source_dataset+'/Lensed_Dataset/'    
        image = matplotlib.image.imread(folder_temp + str(i)+'.png')
        image=image[:,:,0:3]
        R=image[:,:,0]
        G=image[:,:,1]
        B=image[:,:,2]
        lensed_data= 0.2125* R + 0.7154* G + 0.0721 *B
        
        image_1d = util.image2array(lensed_data)  
        # decompose image and return the shapelet coefficients
        coeff_ngc = shapeletSet.decomposition(image_1d, x, y, nmax, beta, deltaPix=deltaPix, center_x=0, center_y=0) 
        # reconstruct the image with the shapelet coefficients
        image_reconstructed = shapeletSet.function(x, y, coeff_ngc, nmax, beta, center_x=0, center_y=0)
        # turn 1d array back into 2d image
        image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image
        image_reconstructed_2d=image_reconstructed_2d/np.max(image_reconstructed_2d)
        # save in the wanted folder
        matplotlib.image.imsave(source_reconstruction+'/Lensed_Dataset/'+str(i)+'.png',image_reconstructed_2d,vmin=np.min(image_reconstructed_2d),vmax=np.max(image_reconstructed_2d))
        lensed_coeff_dataset.append(np.array(coeff_ngc))
        
        ##Noisy data reconstruction
        folder_temp=source_dataset+'/Noisy_Dataset/'    
        image = matplotlib.image.imread(folder_temp + str(i)+'.png')
        image=image[:,:,0:3]
        R=image[:,:,0]
        G=image[:,:,1]
        B=image[:,:,2]
        noisy_data= 0.2125* R + 0.7154* G + 0.0721 *B
        
        image_1d = util.image2array(noisy_data)  
        # decompose image and return the shapelet coefficients
        coeff_ngc = shapeletSet.decomposition(image_1d, x, y, nmax, beta, deltaPix=deltaPix, center_x=0, center_y=0) 
        # reconstruct the image with the shapelet coefficients
        image_reconstructed = shapeletSet.function(x, y, coeff_ngc, nmax, beta, center_x=0, center_y=0)
        # turn 1d array back into 2d image
        image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image
        image_reconstructed_2d=image_reconstructed_2d/np.max(image_reconstructed_2d)
        # save in the wanted folder
        matplotlib.image.imsave(source_reconstruction+'/Noisy_Dataset/'+str(i)+'.png',image_reconstructed_2d,vmin=np.min(image_reconstructed_2d),vmax=np.max(image_reconstructed_2d))
        noisy_coeff_dataset.append(np.array(coeff_ngc))
        
        
      #unlensed_dataset_reconstructed=np.array(unlensed_dataset_reconstructed)
      unlensed_coeff_dataset=np.array(unlensed_coeff_dataset)
    
      #lensed_dataset_reconstructed=np.array(lensed_dataset_reconstructed)
      lensed_coeff_dataset=np.array(lensed_coeff_dataset)
    
      #noisy_dataset_reconstructed=np.array(noisy_dataset_reconstructed)
      noisy_coeff_dataset=np.array(noisy_coeff_dataset)
      
      
      dataFrame_unlensed_coeff_dataset = pd.DataFrame(unlensed_coeff_dataset)
      dataFrame_unlensed_coeff_dataset.columns = np.linspace(0,len(coeff_ngc)-1,num=len(coeff_ngc),dtype=int)
      dataFrame_unlensed_coeff_dataset.to_csv(source_reconstruction+'/DataFrame_unlensed_coeff_dataset/Unlensed_coeff_dataset_DataFrame.csv')
      
      dataFrame_lensed_coeff_dataset = pd.DataFrame(lensed_coeff_dataset)
      dataFrame_lensed_coeff_dataset.columns = np.linspace(0,len(coeff_ngc)-1,num=len(coeff_ngc),dtype=int)
      dataFrame_lensed_coeff_dataset.to_csv(source_reconstruction+'/DataFrame_lensed_coeff_dataset/Lensed_coeff_dataset_DataFrame.csv')
      
      dataFrame_noisy_coeff_dataset = pd.DataFrame(noisy_coeff_dataset)
      dataFrame_noisy_coeff_dataset.columns = np.linspace(0,len(coeff_ngc)-1,num=len(coeff_ngc),dtype=int)
      dataFrame_noisy_coeff_dataset.to_csv(source_reconstruction+'/DataFrame_noisy_coeff_dataset/Noisy_coeff_dataset_DataFrame.csv')
      
      
    
source_dataset='C:/Users/Ayoub/Desktop/PFE/Generation_Dataset/Dataset'

if not os.path.exists(source_dataset):
    generating_dataset(n,source_dataset, data_class, psf_class, kwargs_numerics, exp_time, background_rms, x, y)#generating_dataset(n, data_class, psf_class, kwargs_numerics, exp_time, background_rms, x, y)#generating_dataset(n,data_class,psf_class,kwargs_numerics,exp_time,background_rms,x,y)


source_reconstruction='C:/Users/Ayoub/Desktop/PFE/Generation_Dataset/Shapelet_dataset'
n_=n
choice_type_shapelet=ShapeletSet()



theta_min=0.75
theta_max=128

beta=np.sqrt(theta_max*theta_min)
nmax=int(theta_max/theta_min-1)

print('\nbeta = '+str(beta))
print('nmax = '+str(nmax))


x,y=util.make_grid(numPix=theta_max,deltapix=theta_min)
delta=1


shapelet_reconstruction(n,source_dataset,source_reconstruction,choice_type_shapelet,beta,nmax,x,y,deltaPix)
