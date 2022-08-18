from Lens_Source_Models import lens_model__,source_model__
from import_and_installations import *


# data specifics
background_rms = .005  #  background noise per pixel
exp_time = 500.  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
numPix =128  #  cutout pixel size
deltaPix =0.05  #  pixel size in arcsec (area per pixel = deltaPix**2)
fwhm = 0.05  # full width half max of PSF
psf_type = 'GAUSSIAN'  # 'GAUSSIAN', 'PIXEL', 'NONE'

# generate the coordinate grid and image properties
kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms)
data_class = ImageData(**kwargs_data)

# generate the psf variables
kwargs_psf = {'psf_type': psf_type, 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 3}
psf_class = PSF(**kwargs_psf)
kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}



def generating_dataset(n,folder_path_dataset,data_class,psf_class,kwargs_numerics,exp_time,background_rms,x, y):
  
  variable_dataset=False      
  if not os.path.exists(folder_path_dataset):
      os.makedirs(folder_path_dataset)
      
      
      variable_dataset=True
  
  if variable_dataset:
      ## No light model for the lens
      lens_light_model_list = []
      kwargs_lens_light_list = []
      
      ## Initialization of dataset
      unlensed_dataset=[]
      lensed_dataset=[]
      noisy_dataset=[]
      
      lens_parameters_dataset=[]
      source_type_dataset=[]
      source_parameters_dataset=[]
      ## Type of sources used in the dataset
      source_type_list=['SERSIC_ELLIPSE','HERNQUIST','NIE','CHAMELEON']#,'POWER_LAW']#,'GAUSSIAN_ELLIPSE']#MULTI_GAUSSIAN_ELLIPSE'],'UNIFORM','ELLIPSOID']#,'POWER_LAW',,'UNIFORM','GAUSSIAN_ELLIPSE',,'ELLIPSOID',
      
      ## Type of Lens used in the dataset
      lens_type_list=['EPL']#['EPL','SHEAR','POINT_MASS','SIS','NIE','SIE','SERSIC']#',CORED_DENSITY',,'CURVED_ARC_SIS_MST']
      
      ## number of sources
      nb_source=len(source_type_list)
      
      ##
      Maxim_global=0
      
      for i in tqdm(range(n)):
        source_model_list=[]
        kwargs_source=[]
        
        ## Choice of number of sources
        choix_nb_source=np.random.randint(5,11)
        L_choix_source=np.random.randint(0,nb_source,choix_nb_source)
    
        ## Parameters of each sources
        for k in L_choix_source:
          source_type=source_type_list[k]
          amp_s=uniform(200,600)
          e1_s=uniform(-0.3,0.3)
          e2_s=uniform(-0.3,0.3) 
          center_x=uniform(-2.5,2.5)
          center_y=uniform(-2.5,2.5)
    
          source_model_list.append(source_type)
          kwargs_source_type=source_model__(source_type,amp_s,e1_s,e2_s,center_x,center_y)
          kwargs_source.append(kwargs_source_type)
    
        source_model_class = LightModel(source_model_list)
        kwargs_label_source=[source_model_list,kwargs_source]    
        
        ## Lens_parameters
        gamma=uniform(-0.5,0.5)
        e1=uniform(-0.3,0.3)
        e2=uniform(-0.3,0.3)
        theta_E=uniform(0.3,0.8)
           
        lens_type='EPL'   
        lens_model_list = [lens_type]
        lens_model_class = LensModel(lens_model_list)
    
        kwargs_lens_type=lens_model__(theta_E,gamma,e1,e2)
        kwargs_lens=[kwargs_lens_type]
    
        kwargs_label_lens=[lens_model_list,kwargs_lens]
        
        
        ### Generation DATA
        
        # image without noise
        imageModel = ImageModel(data_class, psf_class, lens_model_class=lens_model_class, source_model_class=source_model_class, kwargs_numerics=kwargs_numerics)#, lens_light_model_class=None,point_source_class=None)
        image_model = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None, source_add=True,lens_light_add=False,  point_source_add=False)
        
        lensed_dataset.append(np.array(image_model))
        
        # noisy image
        poisson = image_util.add_poisson(image_model, exp_time=exp_time)
        bkg = image_util.add_background(image_model, sigma_bkd=background_rms)
        image_real = image_model + poisson + bkg
        
        noisy_dataset.append(np.array(image_real))
        
        # unlensed image
        imageModel_u = ImageModel(data_class, psf_class, kwargs_numerics=kwargs_numerics,source_model_class=source_model_class)#,lens_model_class=None)#, lens_light_model_class=None,point_source_class=None)
        unlensed_image = imageModel_u.image(kwargs_lens=None, kwargs_source=kwargs_source, kwargs_lens_light=None, kwargs_ps=None, source_add=True, lens_light_add=False,point_source_add=False)
        
        unlensed_dataset.append(np.array(unlensed_image))
        
        # Maximum update
        maxim_local=np.max((unlensed_image,image_model))
        Maxim_global=np.max((maxim_local,Maxim_global))


        lens_parameters_dataset.append(np.array([theta_E,gamma,0,0,e1,e2]))
        source_type_dataset.append(source_model_list)
        source_parameters_dataset.append(kwargs_source)
      
      
      unlensed_dataset=np.array(unlensed_dataset)
      lensed_dataset=np.array(lensed_dataset)
      noisy_dataset=np.array(noisy_dataset)
      
      
      lens_parameters_dataset=np.array(lens_parameters_dataset)
      source_type_dataset=np.array(source_type_dataset)
      source_parameters_dataset=np.array(source_parameters_dataset)
      
      ## Same scale to have Relative brightness between images for the training
      noisy_dataset=noisy_dataset/Maxim_global
      lensed_dataset=lensed_dataset/Maxim_global
      unlensed_dataset=unlensed_dataset/Maxim_global
      
      ## Save
      np.save(folder_path_dataset+'/unlensed_dataset'+'.npy', unlensed_dataset)
      np.save(folder_path_dataset+'/lensed_dataset'+'.npy', lensed_dataset)
      np.save(folder_path_dataset+'/noisy_dataset'+'.npy', noisy_dataset)
      
      
      np.save(folder_path_dataset+'/lens_parameters.npy',lens_parameters_dataset)
      np.save(folder_path_dataset+'/source_types.npy',source_type_dataset)
      np.save(folder_path_dataset+'/source_parameters.npy',source_parameters_dataset)
      
     

folder_path_dataset='C:/Users/Ayoub/Desktop/PFE/Generation_Dataset/Dataset'

numPix = 128  #  cutout pixel size
deltaPix =0.05  #  pixel size in arcsec (area per pixel = deltaPix**2)

x, y = util.make_grid(numPix, deltaPix)
n=10

generating_dataset(n,folder_path_dataset, data_class, psf_class, kwargs_numerics, exp_time, background_rms, x, y)#generating_dataset(n, data_class, psf_class, kwargs_numerics, exp_time, background_rms, x, y)#generating_dataset(n,data_class,psf_class,kwargs_numerics,exp_time,background_rms,x,y)

