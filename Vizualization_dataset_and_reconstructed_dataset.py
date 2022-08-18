from import_and_installations import *
from Dataset_generation import generating_dataset,x,y,numPix,deltaPix,data_class, psf_class, kwargs_numerics, exp_time, background_rms,folder_path_dataset,n
from Shapelet_Reconstruction import folder_path_reconstruction,shapelet_reconstruction,choice_type_shapelet,beta,nmax,delta
from Vizualization_shapelets_basis import position,coord_to_pos

      
if not os.path.exists(folder_path_dataset):
    generating_dataset(n,folder_path_dataset, data_class, psf_class, kwargs_numerics, exp_time, background_rms, x, y)

## Image Dataset
filename='unlensed_dataset.npy'
unlensed_dataset=np.load(folder_path_dataset+'/'+filename)

filename='lensed_dataset.npy'
lensed_dataset=np.load(folder_path_dataset+'/'+filename)

filename='noisy_dataset.npy'
noisy_dataset=np.load(folder_path_dataset+'/'+filename)

## Label Datasets
filename='lens_parameters.npy'
lens_parameters_dataset=np.load(folder_path_dataset+'/'+filename)

filename='source_types.npy'
source_type_dataset=np.load(folder_path_dataset+'/'+filename,allow_pickle=True)

filename='source_parameters.npy'
source_parameters_dataset=np.load(folder_path_dataset+'/'+filename,allow_pickle=True)



if not os.path.exists(folder_path_reconstruction):
    shapelet_reconstruction(n,folder_path_dataset,folder_path_reconstruction,choice_type_shapelet,beta,nmax,x,y,delta)

# Shapelet Reconstruction Datasets
## Image Dataset
filename='unlensed_dataset_reconstructed_images.npy'
unlensed_dataset_reconstructed=np.load(folder_path_reconstruction+'/'+filename)

filename='lensed_dataset_reconstructed_images.npy'
lensed_dataset_reconstructed=np.load(folder_path_reconstruction+'/'+filename)

filename='noisy_dataset_reconstructed_images.npy'
noisy_dataset_reconstructed=np.load(folder_path_reconstruction+'/'+filename)
## Shapelet Coeff Dataset
filename='unlensed_dataset_reconstructed_coeff.npy'
unlensed_dataset_reconstructed_coeff=np.load(folder_path_reconstruction+'/'+filename)

filename='lensed_dataset_reconstructed_coeff.npy'
lensed_dataset_reconstructed_coeff=np.load(folder_path_reconstruction+'/'+filename)

filename='noisy_dataset_reconstructed_coeff.npy'
noisy_dataset_reconstructed_coeff=np.load(folder_path_reconstruction+'/'+filename)



## For the Lens Model Plot we need the position of the sources, we have it in the labelset
def coord_source_pos(source_parameters_dataset,n):
  L_sourcePos_x=[]
  L_sourcePos_y=[]

  for ind in range(n):
    label_source=source_parameters_dataset[ind]
    sourcePos_x=[]
    sourcePos_y=[]
    for i in range(len(label_source)):
      center_x=label_source[i]["center_x"]
      center_y=label_source[i]["center_y"]
      sourcePos_x.append(center_x)
      sourcePos_y.append(center_y)
    L_sourcePos_x.append(sourcePos_x)
    L_sourcePos_y.append(sourcePos_y)

  return L_sourcePos_x,L_sourcePos_y


def vizualize_dataset(L_affichage,unlensed_dataset,lensed_dataset,noisy_dataset,lens_parameters_dataset,source_parameters_dataset,n,nb_affichage,numPix,deltaPix):
  numPix,deltaPix=numPix,deltaPix
  
  lens_type='EPL'   
  lens_model_list = [lens_type]
  lensModel=LensModel(lens_model_list)

  L_sourcePos_x,L_sourcePos_y=coord_source_pos(source_parameters_dataset,n)
  
  i=0
  nb_ligne=len(L_affichage)
  f, axes = plt.subplots(nb_ligne, 4, figsize=(2*nb_ligne, 6*nb_ligne), sharex=False, sharey=False)
  for ind in L_affichage:
    
    # sequence of weak lensing
    ax = axes[i][0]
    im = ax.matshow(unlensed_dataset[ind], origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,   extent=[0, 1, 0, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Unlensed Image')
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')
      
    ax = axes[i][1]
    im = ax.matshow(lensed_dataset[ind], origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,  extent=[0, 1, 0, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Lensed Image')
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')
    

    ax = axes[i][2]
    im = ax.matshow(noisy_dataset[ind], origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,  extent=[0, 1, 0, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Noisy image')
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')

    
    sourcePos_x=L_sourcePos_x[ind]
    sourcePos_y=L_sourcePos_y[ind]
    label_lens=lens_parameters_dataset[ind]
    kwargs_lens=[{'theta_E': label_lens[0], 'gamma': label_lens[1], 'center_x': label_lens[2], 'center_y': label_lens[3], 'e1': label_lens[4], 'e2':label_lens[5] }]
    ax=axes[i][3]
    lens_plot.lens_model_plot(ax,lensModel,kwargs_lens,numPix,deltaPix,sourcePos_x,sourcePos_y,with_caustics=True,with_convergence=True)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Lens Model PEMD')
    
    i=i+1

  plt.show()


def vizualize_dataset_shapelet(L_affichage,unlensed_dataset_reconstructed,lensed_dataset_reconstructed,noisy_dataset_reconstructed,n,nb_affichage):
  
 
  nb_ligne=len(L_affichage)
  i=0
  f, axes = plt.subplots(nb_ligne, 3, figsize=(2*nb_ligne, 6*nb_ligne), sharex=False, sharey=False)
  for ind in L_affichage:
    # sequence of weak lensing
    ax = axes[i][0]
    im = ax.matshow(unlensed_dataset_reconstructed[ind], origin='lower')#, vmin=v_min, vmax=v_max, cmap=cmap,  extent=[0, 1, 0, 1]) #vmin=v_min, vmax=v_max, cmap=cmap,
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('Unlensed image')
    ax.autoscale(False)
    
    ax = axes[i][1]
    im = ax.matshow(lensed_dataset_reconstructed[ind], origin='lower')#, vmin=v_min, vmax=v_max, cmap=cmap, extent=[0, 1, 0, 1]) #vmin=v_min, vmax=v_max, cmap=cmap,
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('Lensed image')
    ax.autoscale(False)

    ax = axes[i][2]
    im = ax.matshow(noisy_dataset_reconstructed[ind], origin='lower')#, vmin=v_min, vmax=v_max, cmap=cmap, extent=[0, 1, 0, 1]) #vmin=v_min, vmax=v_max, cmap=cmap,
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('Noisy image')
    ax.autoscale(False)
    
    i=i+1

  plt.show()


def vizualize_dataset_coeff_shapelet(len_coeff_shapelet,unlensed_dataset_reconstructed_coeff,lensed_dataset_reconstructed_coeff,noisy_dataset_reconstructed_coeff,L_affichage,n,nb_affichage):
  
  x_ax_shapelett=np.linspace(0,len_coeff_shapelet-1,num=len_coeff_shapelet,dtype='int')

  nb_ligne=len(L_affichage)
  i=0
  f, axes = plt.subplots(nb_ligne, 3, figsize=(2*nb_ligne, 6*nb_ligne), sharex=False, sharey=False)
  for ind in L_affichage:
    # sequence of weak lensing
    ax = axes[i][0]
    ax.plot(x_ax_shapelett,unlensed_dataset_reconstructed_coeff[ind])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_title('Unlensed image')
    
    ax = axes[i][1]
    ax.plot(x_ax_shapelett,lensed_dataset_reconstructed_coeff[ind])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_title('Lensed image')
    
    ax = axes[i][2]
    ax.plot(x_ax_shapelett,noisy_dataset_reconstructed_coeff[ind])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_title('Noisy image')
    
    i=i+1

  plt.show()
  
#########################################################################################################################################################################
#########################################################################################################################################################################
## Plot Dataset  
numPix=numPix # numPix=number of pixels
deltaPix=deltaPix # deltaPix=size of a pixel (in arcsec)

##n is the number of data
n=n
## nb_affichage is the number of images plotted
nb_affichage=np.minimum(5,n)

L_affichage=np.linspace(0,n-1,num=nb_affichage,dtype='int')
print('indices of data plotted = '+str(L_affichage))


## Plot Images 
vizualize_dataset(L_affichage,unlensed_dataset,lensed_dataset,noisy_dataset,lens_parameters_dataset,source_parameters_dataset,n,nb_affichage,numPix,deltaPix)


## Plot Reconstructed Images with Shapelets
vizualize_dataset_shapelet(L_affichage,unlensed_dataset_reconstructed,lensed_dataset_reconstructed,noisy_dataset_reconstructed,n,nb_affichage)

## Plot Distribution coeff shapelets
len_coeff_shapelet=len(unlensed_dataset_reconstructed_coeff[0])
vizualize_dataset_coeff_shapelet(len_coeff_shapelet,unlensed_dataset_reconstructed_coeff,lensed_dataset_reconstructed_coeff,noisy_dataset_reconstructed_coeff,L_affichage,n,nb_affichage)
