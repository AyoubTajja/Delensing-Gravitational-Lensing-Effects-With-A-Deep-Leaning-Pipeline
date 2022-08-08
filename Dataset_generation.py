#import import_and_installations
from Lens_Source_Models import lens_model__,source_model__
from import_and_installations import *


# data specifics
background_rms = .005  #  background noise per pixel
exp_time = 500.  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
numPix = 128  #  cutout pixel size
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



def generating_dataset(n,data_class,psf_class,kwargs_numerics,exp_time,background_rms,x, y):
  ## No light model for the lens
  lens_light_model_list = []
  kwargs_lens_light_list = []
  
  ## Initialization of dataset and labelset lists
  dataset=[]
  labelset=[]
  
  ## Type of sources used in the dataset
  source_type_list=['NIE','SERSIC_ELLIPSE','HERNQUIST','CHAMELEON','POWER_LAW']#,'GAUSSIAN_ELLIPSE']#MULTI_GAUSSIAN_ELLIPSE'],'UNIFORM','ELLIPSOID']#,'POWER_LAW',,'UNIFORM','GAUSSIAN_ELLIPSE',,'ELLIPSOID',
  
  ## Type of Lens used in the dataset
  lens_type_list=['EPL']#['EPL','SHEAR','POINT_MASS','SIS','NIE','SIE','SERSIC']#',CORED_DENSITY',,'CURVED_ARC_SIS_MST']
  
  ## number of sources
  nb_source=len(source_type_list)

  
  for i in range(n):
    ## For each data in the dataset we have a certain number of sources (between 5 and 10)
    
    ## Initialization of the sources list of the data
    source_model_list=[]
    kwargs_source=[]
    
    ## Choice of number of sources
    choix_nb_source=np.random.randint(5,11)
    L_choix_source=np.random.randint(0,nb_source,choix_nb_source)

    ## Parameters of each sources
    for k in L_choix_source:
      source_type=source_type_list[k]
      amp_s=uniform(200,600)
      e1_s=uniform(-0.2,0.2)
      e2_s=uniform(-0.2,0.2)
      center_x=uniform(-2.5,2.5)
      center_y=uniform(-2.5,2.5)

      source_model_list.append(source_type)
      kwargs_source_type=source_model__(source_type,amp_s,e1_s,e2_s,center_x,center_y)
      kwargs_source.append(kwargs_source_type)

    source_model_class = LightModel(source_model_list)
    kwargs_label_source=[source_model_list,kwargs_source]    
    
    ## Lens_parameters
    gamma=uniform(-1,1)
    e1=uniform(-0.2,0.2)
    e2=uniform(-0.2,0.2)
    theta_E=uniform(0.1,0.9)
       
    lens_type='EPL'   
    lens_model_list = [lens_type]
    lens_model_class = LensModel(lens_model_list)

    kwargs_lens_type=lens_model__(theta_E,gamma,e1,e2)
    kwargs_lens=[kwargs_lens_type]
    
    kwargs_label_lens=[lens_model_list,kwargs_lens]
    
    ### Generation DATA
    imageModel = ImageModel(data_class, psf_class, lens_model_class=lens_model_class, source_model_class=source_model_class, kwargs_numerics=kwargs_numerics)#, lens_light_model_class=None,point_source_class=None)
    # image without noise
    image_model = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None, source_add=True,lens_light_add=False,  point_source_add=False)
    # noisy image
    poisson = image_util.add_poisson(image_model, exp_time=exp_time)
    bkg = image_util.add_background(image_model, sigma_bkd=background_rms)
    image_real = image_model + poisson + bkg
    # unlensed image
    imageModel_u = ImageModel(data_class, psf_class, kwargs_numerics=kwargs_numerics,source_model_class=source_model_class)#,lens_model_class=None)#, lens_light_model_class=None,point_source_class=None)
    unlensed_image = imageModel_u.image(kwargs_lens=None, kwargs_source=kwargs_source, kwargs_lens_light=None, kwargs_ps=None, source_add=True, lens_light_add=False,point_source_add=False)

    local_data=[unlensed_image,image_model,image_real]

    kwargs_label=[kwargs_label_source,kwargs_label_lens]
    local_label=kwargs_label
    
    dataset.append(np.array(local_data))
    labelset.append(np.array(local_label))

  dataset=np.array(dataset)  
  labelset=np.array(labelset)
  return dataset,labelset



x, y = util.make_grid(numPix, deltaPix)
n=100

dataset,labelset=generating_dataset(n,data_class,psf_class,kwargs_numerics,exp_time,background_rms,x,y)
