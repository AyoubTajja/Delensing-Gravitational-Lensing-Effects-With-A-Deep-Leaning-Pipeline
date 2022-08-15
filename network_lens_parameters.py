from import_and_installations import *
        
model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3),input_shape=(128, 128, 1),use_bias=False))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.BatchNormalization())
model1.add(layers.Activation("relu"))
model1.add(layers.Conv2D(64, (3, 3) ,use_bias=False))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.BatchNormalization())
model1.add(layers.Activation("relu"))
model1.add(layers.Conv2D(64, (3, 3),use_bias=False))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.BatchNormalization())
model1.add(layers.Activation("relu"))
model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(6,activation='linear'))

