from import_and_installations import *
from Shapelet_Reconstruction import beta,theta_min,theta_max
from Dataset_generation import numPix,deltaPix


## Function that from the N position in the shapelet vector decomposition gives the n1,n2 integers of the shapelet
def position(N):
    res=0
    if N<0:
      return None,None
    res=int((-3+np.sqrt(1+8*N))/2)+1
    pos_x=res-(N-res*(res+1)/2)
    pos_y=res-pos_x
    
    return (int(pos_x),int(pos_y))

## Function that from the n1,n2 integers of the shapelet gives the N position in the shapelet vector decomposition
def coord_to_pos(n1,n2):
  if (n1<0 or n2<0):
    return None
  else :
    n=n1+n2
    N=n2+((n+1)*n/2)
    return int(N)

# Function that enables plot
def nb_ligne_col(n_shap):
  if n_shap%5==0:
    nb_ligne=n_shap//5
    nb_colonne=5
    return nb_ligne,nb_colonne
  elif n_shap%4==0:
    nb_ligne=n_shap//4
    nb_colonne=4
    return nb_ligne,nb_colonne
  elif n_shap%3==0:
    nb_ligne=n_shap//3
    nb_colonne=3
    return nb_ligne,nb_colonne
  else :
    return n_shap,0


numPix=theta_max
deltaPix=theta_min


## choice of the nmax to plot some basis shapelets 
n_max_plot=8

## Choice of the type of shapelets : ShapeletSet() is cartesian and ShapeletSetPolar() is polar
shapeletSet=ShapeletSet()

# vector that contains the shapelet basis
shapelet_basis=shapeletSet.shapelet_basis_2d(n_max_plot, beta, numPix, deltaPix=deltaPix, center_x=0, center_y=0)
print('len(shapelet_basis) = '+str(len(shapelet_basis)))

n_shap=len(shapelet_basis)


nb_ligne, nb_colonne=nb_ligne_col(n_shap)

f, axes = plt.subplots(nb_ligne, nb_colonne, figsize=(14,20), sharex=False, sharey=False)
for i in range(nb_ligne):
  for j in range(nb_colonne):
    ax = axes[i][j]
    im = ax.matshow(shapelet_basis[j+i*nb_colonne], origin='lower')
    ax.set_title(position(j+i*nb_colonne))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')
