# import
import keras

# Step 1: Defining the callbacks
#------------------------------
# define early stopping callback
earlystopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# define model checkpoint callback
checkpoint = keras.callbacks.ModelCheckpoint("checkpoint_model_{epoch:02d}.hdf5", monitor='loss', verbose=1, save_best_only=False, mode='auto', period=10)

# Step 2: Assigning the callback
#------------------------------
# fit the model
history = model.fit(train_data_gen, 
#                     ....
                    callbacks=[checkpoint, earlystopping]
                   )
