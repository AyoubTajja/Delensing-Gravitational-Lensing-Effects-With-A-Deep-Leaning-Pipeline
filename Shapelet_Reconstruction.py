from import_and_installations import *
from Dataset_generation import generating_dataset,x,y,numPix,deltaPix,data_class, psf_class, kwargs_numerics, exp_time, background_rms,folder_path_dataset,n



def shapelet_reconstruction(n,folder_path_dataset,folder_path_reconstruction,choice_type_shapelet,beta,nmax,x,y,deltaPix):
  variable_dataset=False
    
  if not os.path.exists(folder_path_reconstruction):
    os.makedirs(folder_path_reconstruction)
    
    ## Image Dataset
    
    filename='unlensed_dataset.npy'
    unlensed_dataset=np.load(folder_path_dataset+'/'+filename)
    
    filename='lensed_dataset.npy'
    lensed_dataset=np.load(folder_path_dataset+'/'+filename)
    
    filename='noisy_dataset.npy'
    noisy_dataset=np.load(folder_path_dataset+'/'+filename)

    variable_dataset=True
    
  if variable_dataset:
      
      unlensed_dataset_reconstructed=[]
      lensed_dataset_reconstructed=[]
      noisy_dataset_reconstructed=[]
      
      unlensed_coeff_dataset=[]
      lensed_coeff_dataset=[]
      noisy_coeff_dataset=[]
     
      ## Choose the type of shapelets : Polar or Cartesian shapelets
      shapeletSet = choice_type_shapelet
      
       
      for i in tqdm(range(n)):
        '''  
        ##Unlensed data reconstruction
        folder_temp=source_dataset+'/Unlensed_Dataset/'    
        image = matplotlib.image.imread(folder_temp + str(i)+'.png')
        image=image[:,:,0:3]
        R=image[:,:,0]
        G=image[:,:,1]
        B=image[:,:,2]
        unlensed_data= 0.2125* R + 0.7154* G + 0.0721 *B
        '''
        data=unlensed_dataset[i]
        image_1d = util.image2array(data)
        coeff_ngc = shapeletSet.decomposition(image_1d, x, y, nmax, beta, deltaPix=deltaPix, center_x=0, center_y=0) 
        # reconstruct the image with the shapelet coefficients
        image_reconstructed = shapeletSet.function(x, y, coeff_ngc, nmax, beta, center_x=0, center_y=0)
        # turn 1d array back into 2d image
        image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image
        #image_reconstructed_2d=image_reconstructed_2d/np.max(image_reconstructed_2d)
        # save in the wanted folder
        #matplotlib.image.imsave(source_reconstruction+'/Unlensed_Dataset/'+str(i)+'.png',image_reconstructed_2d,vmin=np.min(image_reconstructed_2d),vmax=np.max(image_reconstructed_2d))
        unlensed_dataset_reconstructed.append(np.array(image_reconstructed_2d))
        unlensed_coeff_dataset.append(np.array(coeff_ngc))
        
        '''
        ##Lensed data reconstruction
        folder_temp=source_dataset+'/Lensed_Dataset/'    
        image = matplotlib.image.imread(folder_temp + str(i)+'.png')
        image=image[:,:,0:3]
        R=image[:,:,0]
        G=image[:,:,1]
        B=image[:,:,2]
        lensed_data= 0.2125* R + 0.7154* G + 0.0721 *B
        '''
        data=lensed_dataset[i]
        image_1d = util.image2array(data)  
        # decompose image and return the shapelet coefficients
        coeff_ngc = shapeletSet.decomposition(image_1d, x, y, nmax, beta, deltaPix=deltaPix, center_x=0, center_y=0) 
        # reconstruct the image with the shapelet coefficients
        image_reconstructed = shapeletSet.function(x, y, coeff_ngc, nmax, beta, center_x=0, center_y=0)
        # turn 1d array back into 2d image
        image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image
        #image_reconstructed_2d=image_reconstructed_2d/np.max(image_reconstructed_2d)
        # save in the wanted folder
        #matplotlib.image.imsave(source_reconstruction+'/Lensed_Dataset/'+str(i)+'.png',image_reconstructed_2d,vmin=np.min(image_reconstructed_2d),vmax=np.max(image_reconstructed_2d))
        lensed_dataset_reconstructed.append(np.array(image_reconstructed_2d))
        lensed_coeff_dataset.append(np.array(coeff_ngc))
        
        '''
        ##Noisy data reconstruction
        folder_temp=source_dataset+'/Noisy_Dataset/'    
        image = matplotlib.image.imread(folder_temp + str(i)+'.png')
        image=image[:,:,0:3]
        R=image[:,:,0]
        G=image[:,:,1]
        B=image[:,:,2]
        noisy_data= 0.2125* R + 0.7154* G + 0.0721 *B
        '''
        data=noisy_dataset[i]
        image_1d = util.image2array(data)  
        # decompose image and return the shapelet coefficients
        coeff_ngc = shapeletSet.decomposition(image_1d, x, y, nmax, beta, deltaPix=deltaPix, center_x=0, center_y=0) 
        # reconstruct the image with the shapelet coefficients
        image_reconstructed = shapeletSet.function(x, y, coeff_ngc, nmax, beta, center_x=0, center_y=0)
        # turn 1d array back into 2d image
        image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image
        #image_reconstructed_2d=image_reconstructed_2d/np.max(image_reconstructed_2d)
        # save in the wanted folder
        #matplotlib.image.imsave(source_reconstruction+'/Noisy_Dataset/'+str(i)+'.png',image_reconstructed_2d,vmin=np.min(image_reconstructed_2d),vmax=np.max(image_reconstructed_2d))
        noisy_dataset_reconstructed.append(np.array(image_reconstructed_2d))
        noisy_coeff_dataset.append(np.array(coeff_ngc))
        
        
      unlensed_dataset_reconstructed=np.array(unlensed_dataset_reconstructed)
      unlensed_coeff_dataset=np.array(unlensed_coeff_dataset)
    
      lensed_dataset_reconstructed=np.array(lensed_dataset_reconstructed)
      lensed_coeff_dataset=np.array(lensed_coeff_dataset)
    
      noisy_dataset_reconstructed=np.array(noisy_dataset_reconstructed)
      noisy_coeff_dataset=np.array(noisy_coeff_dataset)
      
      print(len(coeff_ngc))
      ## Save
      np.save(folder_path_reconstruction+'/unlensed_dataset_reconstructed_images'+'.npy', unlensed_dataset_reconstructed)
      np.save(folder_path_reconstruction+'/lensed_dataset_reconstructed_images'+'.npy', lensed_dataset_reconstructed)
      np.save(folder_path_reconstruction+'/noisy_dataset_reconstructed_images'+'.npy', noisy_dataset_reconstructed)
      
      
      np.save(folder_path_reconstruction+'/unlensed_dataset_reconstructed_coeff'+'.npy', unlensed_coeff_dataset)
      np.save(folder_path_reconstruction+'/lensed_dataset_reconstructed_coeff'+'.npy', lensed_coeff_dataset)
      np.save(folder_path_reconstruction+'/noisy_dataset_reconstructed_coeff'+'.npy', noisy_coeff_dataset)
      
      
      
if not os.path.exists(folder_path_dataset):
    generating_dataset(n,folder_path_dataset, data_class, psf_class, kwargs_numerics, exp_time, background_rms, x, y)#generating_dataset(n, data_class, psf_class, kwargs_numerics, exp_time, background_rms, x, y)#generating_dataset(n,data_class,psf_class,kwargs_numerics,exp_time,background_rms,x,y)


folder_path_reconstruction='C:/Users/Ayoub/Desktop/PFE/Generation_Dataset/Shapelet_dataset'
n_=n
choice_type_shapelet=ShapeletSet()


theta_min=deltaPix
theta_max=numPix


beta=0.28  #np.sqrt(theta_max*theta_min)
nmax=10000  #np.min((170,int(theta_max/theta_min-1)))

print('\nbeta = '+str(beta))
print('nmax = '+str(nmax))


x,y=util.make_grid(numPix=theta_max,deltapix=theta_min)
delta=deltaPix


shapelet_reconstruction(n,folder_path_dataset,folder_path_reconstruction,choice_type_shapelet,beta,nmax,x,y,delta)
