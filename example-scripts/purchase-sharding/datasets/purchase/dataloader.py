import numpy as np
import os

pwd = os.path.dirname(os.path.realpath(__file__))

train_data = np.load(os.path.join(pwd, 'purchase2_train.npy'), allow_pickle=True)
test_data = np.load(os.path.join(pwd, 'purchase2_test.npy'), allow_pickle=True)

train_data = train_data.reshape((1,))[0]
test_data = test_data.reshape((1,))[0]

X_train = train_data['X'].astype(np.float32)
X_test = test_data['X'].astype(np.float32)
y_train = train_data['y'].astype(np.int64)
y_test = test_data['y'].astype(np.int64)

def load(indices, shard=2, category='train'):
    if category == 'train': 
        if shard == 4:
            for idx, element in enumerate(indices):
                if element == y_train.size:
                    indices[idx] = element - 1
                elif element > y_train.size:
                    indices[idx] = element - (element - y_train.size) -1 
        return X_train[indices], y_train[indices] # The index out of bound error is coming from the y_train
        #size of y_train: 249215
    elif category == 'test':
        return X_test[indices], y_test[indices]