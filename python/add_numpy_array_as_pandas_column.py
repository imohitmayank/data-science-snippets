# Taken from: https://stackoverflow.com/questions/18646076/add-numpy-array-as-column-to-pandas-data-frame
import numpy as np
import pandas as pd
import scipy.sparse as sparse

df = pd.DataFrame(np.arange(1,10).reshape(3,3))
arr = sparse.coo_matrix(([1,1,1], ([0,1,2], [1,2,0])), shape=(3,3))
df['newcol'] = arr.toarray().tolist()
print(df)
