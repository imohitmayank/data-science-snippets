"""Use Avg pooling on output of LSTM, RNN, GRU, or any recurrent layer.
Author: Mohit Mayank
Taken from my answer to this question: https://stackoverflow.com/questions/36428323/lstm-followed-by-mean-pooling/64630846#64630846
"""

# create sample data
A=np.array([[1,2,3],[4,5,6],[0,0,0],[0,0,0],[0,0,0]])
B=np.array([[1,3,0],[4,0,0],[0,0,1],[0,0,0],[0,0,0]])
C=np.array([A,B]).astype("float32")
# expected answer (for temporal mean)
np.mean(C, axis=1)
"""
The output is

array([[1. , 1.4, 1.8],
       [1. , 0.6, 0.2]], dtype=float32)
Now using AveragePooling1D,
"""
model = keras.models.Sequential(
        tf.keras.layers.AveragePooling1D(pool_size=5)
)
model.predict(C)
"""
The output is,

array([[[1. , 1.4, 1.8]],
       [[1. , 0.6, 0.2]]], dtype=float32)
Some points to consider,
- The pool_size should be equal to the step/timesteps size of the recurrent layer.
- The shape of the output is (batch_size, downsampled_steps, features), which contains one additional downsampled_steps dimension. This will be always 1 if you set the pool_size equal to timestep size in recurrent layer.
"""
