from import_and_installations import *
from Dataset_generation import dataset,labelset

deltaPix=0.05 ##in arcsec
numPix=128
x, y = util.make_grid(numPix, deltaPix)

## Choice of data
ind=1
unlensed_image=dataset[ind][0]
lensed_image=dataset[ind][1]

# Parameters
EPL_=epl.EPL()
kwargs_lens=labelset[ind][1][1]


## Theta
theta_x=copy.deepcopy(x)
theta_y=copy.deepcopy(y)

theta_x=theta_x.reshape(-1,numPix)
theta_y=theta_y.reshape(-1,numPix)


## Lensing Potential
vector_potential=EPL_.function(theta_x, theta_y, kwargs_lens[0]["theta_E"], kwargs_lens[0]["gamma"], kwargs_lens[0]["e1"], kwargs_lens[0]["e2"],0,0)


potential=vector_potential.reshape(-1,numPix)

### Deflection angle
alpha_x,alpha_y=np.gradient(potential,deltaPix)

#alpha_x,alpha_y=EPL_.derivatives(theta_x, theta_y, kwargs_lens[0]["theta_E"], kwargs_lens[0]["gamma"], kwargs_lens[0]["e1"], kwargs_lens[0]["e2"], 0, 0)
#alpha_x,alpha_y=alpha_x.reshape(-1,numPix),alpha_y.reshape(-1,numPix)


##Lensing equation
beta_x,beta_y=theta_x-alpha_x,theta_y-alpha_y


## Plot Lensing potential, beta, theta and deflection angle alpha

f, ax = plt.subplots(1, 1, figsize=(5, 10), sharex=False, sharey=False)
im = ax.matshow(potential, origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,   extent=[0, 1, 0, 1])
ax.set_title('Lensing Potential')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cax, orientation='vertical')
plt.show()



f, axes = plt.subplots(3, 2, figsize=(10, 15), sharex=False, sharey=False)
ax=axes[0][0]
im = ax.matshow(alpha_x, origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,   extent=[0, 1, 0, 1])
ax.set_title('alpha_x')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cax, orientation='vertical')

ax=axes[0][1]
im = ax.matshow(alpha_y, origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,   extent=[0, 1, 0, 1])
ax.set_title('alpha_y')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cax, orientation='vertical')


ax=axes[1][0]
im = ax.matshow(theta_x, origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,   extent=[0, 1, 0, 1])
ax.set_title('theta_x')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cax, orientation='vertical')

ax=axes[1][1]
im = ax.matshow(theta_y, origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,   extent=[0, 1, 0, 1])
ax.set_title('theta_y')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cax, orientation='vertical')


ax=axes[2][0]
im = ax.matshow(beta_x, origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,   extent=[0, 1, 0, 1])
ax.set_title('beta_x')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cax, orientation='vertical')

ax=axes[2][1]
im = ax.matshow(beta_y, origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,   extent=[0, 1, 0, 1])
ax.set_title('beta_y')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cax, orientation='vertical')

plt.show()


## Reconstruction with Lensing equation

numPix=128
deltaPix=0.05

shift = deltaPix * (numPix - 1) / 2

## Beta in pixel unit
Coord_beta_x=(beta_x+shift)/(deltaPix)
Coord_beta_y=(theta_y+shift)/(deltaPix)
Coord_beta_x=np.array(Coord_beta_x).reshape(-1,numPix)
Coord_beta_y=np.array(Coord_beta_y).reshape(-1,numPix)

## Theta in pixel unit
Coord_theta_x=(theta_x+shift)/(deltaPix)
Coord_theta_y=(theta_y+shift)/(deltaPix)
Coord_theta_x=np.array(Coord_theta_x).reshape(-1,numPix)
Coord_theta_y=np.array(Coord_theta_y).reshape(-1,numPix)


## Comparaison between unlensed_image[beta_x[i][j]][beta_y[i][j]] and lensed_image[theta_x[i][j]][theta_y[i][j]]
Coord_list_x=[]
Coord_list_y=[]
L_=[]
A=np.zeros((numPix,numPix))

for i in range(numPix):
  for j in range(numPix):
    if int(Coord_beta_x[i][j])<128 and int(Coord_beta_x[i][j])>=0:
      if int(Coord_beta_y[i][j])<128 and int(Coord_beta_y[i][j])>=0:
        A[int(Coord_beta_x[i][j])][int(Coord_beta_y[i][j])]=unlensed_image[int(Coord_beta_x[i][j])][int(Coord_beta_y[i][j])]
        Coord_list_x.append(int(Coord_beta_x[i][j]))
        Coord_list_y.append(int(Coord_beta_y[i][j]))
        #print('lensed = '+str(lensed_image[int(Coord_theta_x[i][j])][int(Coord_theta_y[i][j])]))
        #print('unlensed = '+str(unlensed_image[int(Coord_beta_x[i][j])][int(Coord_beta_y[i][j])]))
        L_.append(np.abs(lensed_image[int(Coord_theta_x[i][j])][int(Coord_theta_y[i][j])]-unlensed_image[int(Coord_beta_x[i][j])][int(Coord_beta_y[i][j])]))

plt.plot(L_)
plt.title('The number of pixels where the comparaison is < 0.1 : '+str(np.count_nonzero(np.where(np.abs(np.array(L_))<0.1))))
plt.show()

print(len(L_))

f, axes = plt.subplots(1, 2, figsize=(int(0.8*nb_ligne), 3*nb_ligne), sharex=False, sharey=False)
ax=axes[0]
ax.imshow(np.log10(A))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('Unlensed reconstructed with lens.eq')
ax.autoscale(False)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cax, orientation='vertical')

ax=axes[1]
ax.imshow(np.log10(unlensed_image))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('True Unlensed Image')
ax.autoscale(False)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im, cax=cax, orientation='vertical')
plt.plot()