from import_and_installations import *
from open_dataset import *
from network_lens_parameters import model1
#from network_unlensed_images import model2
from Dataset_generation import generating_dataset,x,y,numPix,deltaPix,data_class, psf_class, kwargs_numerics, exp_time, background_rms,source_dataset,n


if not os.path.exists(source_dataset):
    generating_dataset(n,source_dataset, data_class, psf_class, kwargs_numerics, exp_time, background_rms, x, y)

## Open dataset
# unlensed_dataset is a numpy.array type
folder='Unlensed_Dataset/'
unlensed_dataset=open_dataset(source_dataset+'/'+folder)
# lensed_dataset is a numpy.array type
folder='Lensed_Dataset/'
lensed_dataset=open_dataset(source_dataset+'/'+folder)
# Noisy_dataset is a numpy.array type
folder='Noisy_Dataset/'
Noisy_Dataset_dataset=open_dataset(source_dataset+'/'+folder)


folder='DataFrame_lens_parameters/'
filename = os.listdir(source_dataset+'/'+folder)[0]
lens_parameters_df = pd.read_csv(source_dataset+'/'+folder+filename)

lensed_parameters_dataset=lens_parameters_df.to_numpy()
lensed_parameters_dataset=np.transpose(lensed_parameters_dataset)[1:]
lensed_parameters_dataset=np.transpose(lensed_parameters_dataset)


## Define the set of training,validation,test

x_train = lensed_dataset[0:int(0.75*n)].reshape(-1,128, 128, 1)
trainLabels=lensed_parameters_dataset[0:int(0.75*n)]
x_dev=lensed_dataset[int(0.75*n):int(0.9*n)].reshape(-1,128, 128, 1)
devLabels=lensed_parameters_dataset[int(0.75*n):int(0.9*n)]
x_test=lensed_dataset[int(0.9*n):].reshape(-1,128, 128, 1)
testLabels=lensed_parameters_dataset[int(0.9*n):]



#####################################################################################################
#####################################################################################################

## First DeepLearning Model to find the lensing parameters
## The input : the Lensed images
## The output : the Lensing parameters of each lensed images
model1.summary()

model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])
history = model1.fit(x_train, trainLabels,batch_size=128, epochs=7, validation_data=(x_dev, devLabels))


# Get training and test loss histories
training_loss = history.history['loss']
dev_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, dev_loss, 'b-')
plt.legend(['Training Loss', 'Dev Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

# plot metrics
plt.plot(history.history['mse'])
plt.show()

plt.plot(history.history['mae'])
plt.show()

plt.plot(history.history['mape'])
plt.show()

## Prediction and testing results
test_results = model1.predict(x_test)

test_results_eval = model1.evaluate(x_test, testLabels, verbose=0)

testout = pd.DataFrame(model1.predict(x_test))

testout.columns = ['theta_E_out', 'gamma_out', 'center_x_out', 'center_y_out','e1_out', 'e2_out']
display(testout)

display(lens_parameters_df)


#####################################################################################################
#####################################################################################################

## First DeepLearning Model to find the unlensed images
## The input : the Lensed images
## The output : the unlensed images
model2.summary()
tf.keras.utils.plot_model(model6, show_shapes=True)

model2.compile(loss=mse_ssim_loss, optimizer=optimizer, metrics=['mse',ssim_loss,psnr_loss])
history2 = model2.fit(x_train, x_trainunlensed,batch_size=128, epochs=  10, validation_data=(x_dev, x_devunlensed))

pred_unlensed = model2.predict(x_test)


# Get training and test loss histories
training_loss = history2.history['loss']
dev_loss = history2.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, dev_loss, 'b-')
plt.legend(['Training Loss', 'Dev Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# plot metrics
plt.plot(history2.history['mse'])
plt.show()

plt.plot(history2.history["ssim_loss"])
plt.show()

nb_affichage_=20

f, axes = plt.subplots(3*nb_affichage_,ncols=3, figsize=(12, 4))
for i in range(nb_affichage_):
   plot = True
   if plot:
     
     ax = axes[i][0]
     im = ax.matshow(lensed_dataset[i], origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,   extent=[0, 1, 0, 1])
     ax.get_xaxis().set_visible(False)
     ax.get_yaxis().set_visible(False)
     ax.set_title('Lensed Image')
     ax.autoscale(False)
     divider = make_axes_locatable(ax)
     cax = divider.append_axes('right', size='5%', pad=0.05)
     f.colorbar(im, cax=cax, orientation='vertical')
        
     
     ax = axes[i][2]
     im = ax.matshow(unlensed_dataset[i], origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,   extent=[0, 1, 0, 1])
     ax.get_xaxis().set_visible(False)
     ax.get_yaxis().set_visible(False)
     ax.set_title('True UnLensed')
     ax.autoscale(False)
     divider = make_axes_locatable(ax)
     cax = divider.append_axes('right', size='5%', pad=0.05)
     f.colorbar(im, cax=cax, orientation='vertical')

     ax = axes[i][2]
     im = ax.matshow(pred_unlensed[i], origin='lower')#,vmin=v_min, vmax=v_max, cmap=cmap,   extent=[0, 1, 0, 1])
     ax.get_xaxis().set_visible(False)
     ax.get_yaxis().set_visible(False)
     ax.set_title('Predicted Unlensed')
     ax.autoscale(False)
     divider = make_axes_locatable(ax)
     cax = divider.append_axes('right', size='5%', pad=0.05)
     f.colorbar(im, cax=cax, orientation='vertical')
