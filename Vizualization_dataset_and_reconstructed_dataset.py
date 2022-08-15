from import_and_installations import *
from Dataset_generation import dataset,labelset,n
from Shapelet_Reconstruction import dataset_reconstructed,coeff_dataset
from Vizualization_shapelets_basis import position,coord_to_pos


## Open dataset
source_dataset='C:/Users/Ayoub/Desktop/PFE/Generation_Dataset/Dataset'
folder='Unlensed_Dataset/'
unlensed_dataset=open_dataset(source+'/'+folder)
folder='Lensed_Dataset/'
lensed_dataset=open_dataset(source+'/'+folder)
folder='Noisy_Dataset/'
Noisy_Dataset_dataset=open_dataset(source+'/'+folder)


## For the Lens Model Plot we need the position of the sources, we have it in the labelset
def coord_source_pos(labelset,n):
  L_sourcePos_x=[]
  L_sourcePos_y=[]

  for ind in range(n):
    label_source=labelset[ind][0][1]
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

def vizualize_dataset(L_affichage,nb_source,unlensed_dataset,lensed_dataset,noisy_dataset,labelset,n,nb_affichage,numPix,deltaPix):
  numPix,deltaPix=numPix,deltaPix
  lens_type='EPL'   
  lens_model_list = [lens_type]
  lensModel=LensModel(lens_model_list)

  L_sourcePos_x,L_sourcePos_y=coord_source_pos(labelset,n)
  # display the initial simulated image
  '''
  cmap_string = 'gray'
  cmap = plt.get_cmap(cmap_string)
  cmap.set_bad(color='k', alpha=1.)
  cmap.set_under('k')
  v_min = -4
  v_max = 1
  '''
  i=0
  nb_ligne=len(L_affichage)
  f, axes = plt.subplots(nb_ligne, 4, figsize=(int(0.8*nb_ligne), 3*nb_ligne), sharex=False, sharey=False)
  for ind in L_affichage:
    
    # sequence of weak lensing
    ax = axes[i][0]
    im = ax.matshow(np.log10(unlensed_dataset[ind]), origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,   extent=[0, 1, 0, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Unlensed Image')
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')
      
    ax = axes[i][1]
    im = ax.matshow(np.log10(lensed_dataset[ind]), origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,  extent=[0, 1, 0, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Lensed Image')
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')
    

    ax = axes[i][2]
    im = ax.matshow(np.log10(noisy_dataset[ind]), origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,  extent=[0, 1, 0, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Noisy image')
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')

    
    sourcePos_x=L_sourcePos_x[ind]
    sourcePos_y=L_sourcePos_y[ind]
    kwargs_lens=labelset[ind][1][1]
    ax=axes[i][3]
    lens_plot.lens_model_plot(ax,lensModel,kwargs_lens,numPix,deltaPix,sourcePos_x,sourcePos_y,with_caustics=True,with_convergence=True)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Lens Model PEMD')

    i=i+1

  plt.show()


def vizualize_dataset_shapelet(L_affichage,dataset_reconstructed,n,nb_affichage,nb_source):
  
  # display the initial simulated image
  '''
  cmap_string = 'gray'
  cmap = plt.get_cmap(cmap_string)
  cmap.set_bad(color='k', alpha=1.)
  cmap.set_under('k')
  v_min = -4
  v_max = 1
  '''
  nb_ligne=len(L_affichage)
  i=0
  f, axes = plt.subplots(nb_ligne, 3, figsize=(int(0.5*nb_ligne), 3*nb_ligne), sharex=False, sharey=False)
  for ind in L_affichage:
    # sequence of weak lensing
    ax = axes[i][0]
    im = ax.matshow(np.log10(dataset_reconstructed[ind][0]), origin='lower')#, vmin=v_min, vmax=v_max, cmap=cmap,  extent=[0, 1, 0, 1]) #vmin=v_min, vmax=v_max, cmap=cmap,
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')
    #ax.set_title('Source Image')
    ax.autoscale(False)
    
    ax = axes[i][1]
    im = ax.matshow(np.log10(dataset_reconstructed[ind][1]), origin='lower')#, vmin=v_min, vmax=v_max, cmap=cmap, extent=[0, 1, 0, 1]) #vmin=v_min, vmax=v_max, cmap=cmap,
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')
    #ax.set_title(labelset_ind_s[ind][j-1][0])
    ax.autoscale(False)

    ax = axes[i][2]
    im = ax.matshow(np.log10(dataset_reconstructed[ind][2]), origin='lower')#, vmin=v_min, vmax=v_max, cmap=cmap, extent=[0, 1, 0, 1]) #vmin=v_min, vmax=v_max, cmap=cmap,
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')
    #ax.set_title(labelset_ind_s[ind][j][0])
    ax.autoscale(False)
    
    i=i+1

  plt.show()


def vizualize_dataset_coeff_shapelet(len_coeff_shapelet,L_affichage,coeff_dataset,n,nb_affichage,nb_source):
  # display the initial simulated image
  '''
  cmap_string = 'gray'
  cmap = plt.get_cmap(cmap_string)
  cmap.set_bad(color='k', alpha=1.)
  cmap.set_under('k')
  v_min = -4
  v_max = 1
  '''
  x_ax_shapelett=np.linspace(0,len_coeff_shapelet-1,num=len_coeff_shapelet,dtype='int')

  nb_ligne=len(L_affichage)
  i=0
  f, axes = plt.subplots(nb_ligne, 3, figsize=(int(0.5*nb_ligne), 3*nb_ligne), sharex=False, sharey=False)
  for ind in L_affichage:
    # sequence of weak lensing
    ax = axes[i][0]
    ax.plot(x_ax_shapelett,coeff_dataset[ind][0])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    
    ax = axes[i][1]
    ax.plot(x_ax_shapelett,coeff_dataset[ind][1])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)

    ax = axes[i][2]
    ax.plot(x_ax_shapelett,coeff_dataset[ind][2])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    
    i=i+1

  plt.show()
  
#########################################################################################################################################################################
#########################################################################################################################################################################

## Plot Dataset  
source_type_list=['SERSIC_ELLIPSE','POWER_LAW','ELLIPSOID','HERNQUIST','UNIFORM','GAUSSIAN_ELLIPSE','CHAMALEON','MULTI_GAUSSIAN_ELLIPSE']
nb_source=len(source_type_list)

numPix,deltaPix=128,0.05 ## numPix=number of pixels and deltaPix=size of a pixel (in arcsec)

##n is the number of data
n=n
## nb_affichage is the number of images plotted
nb_affichage=np.minimum(10,n)

L_affichage=np.linspace(0,n-1,num=nb_affichage,dtype='int')
print('indices of data plotted = '+str(L_affichage))

dataset_=dataset
labelset_=labelset

vizualize_dataset(L_affichage,nb_source,dataset,labelset,n,nb_affichage,numPix,deltaPix)

'''
## Plot Reconstructed Images with Shapelets
dataset_reconstructed_=dataset_reconstructed

vizualize_dataset_shapelet(L_affichage,dataset_reconstructed_,n,nb_affichage,nb_source)


## Plot Distribution coeff shapelets
coeff_dataset_=coeff_dataset
len_coeff_shapelet=len(coeff_dataset[0][0])
vizualize_dataset_coeff_shapelet(len_coeff_shapelet,L_affichage,coeff_dataset_,n,nb_affichage,nb_source)
