''' 
Author: Danijel Kivaranovic 
Title: Neural network (Keras) with sparse data
'''

## import libraries
import numpy as np
np.random.seed(123)

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import CSVLogger,EarlyStopping, ModelCheckpoint

## Batch generators ##################################################################################################################################

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

## read data
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

index = list(train.index)
print index[0:10]
np.random.shuffle(index)
print index[0:10]
train = train.iloc[index]
'train = train.iloc[np.random.permutation(len(train))]'

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
y = np.log(train['loss'].values+200)
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

## neural net
def nn_model():
    model = Sequential()
    
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)

## cv-folds
nfolds = 10
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

## train models
i = 0
nbags = 10
nepochs = 55
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

## 10 different bags, each with 10 different CV so as to lower the variance and generalixation error
for j in range(nbags):
    xtrain,y = shuffle(xtrain,y)
    
    import time
    start_time = time.time()
    i = 0
    bag = np.zeros(xtrain.shape[0])
    for (inTr, inTe) in folds:
        xtr = xtrain[inTr]
        ytr = y[inTr]
        xte = xtrain[inTe]
        yte = y[inTe]
        pred = np.zeros(xte.shape[0])
        
        model = nn_model()
        csv_logger = CSVLogger('log'+str(j)+'-'+str(i)+'.txt')
        checkpointer = ModelCheckpoint(filepath='models/Model'+str(j)+'-'+str(i)+'.hdf5', verbose=1, save_best_only=True)
        #earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode='min')   ##will do in bigger models iter
        fit = model.fit_generator(generator = batch_generator(xtr, ytr,128, True),
        			  nb_epoch = nepochs,
        			  validation_data = batch_generator(xte,yte,800,False),
        			  nb_val_samples = xte.shape[0],
        			  callbacks = [csv_logger,checkpointer],
        			  samples_per_epoch = xtr.shape[0],
        			  verbose = 1)
        			  
        model.save('models/Last_Iter'+str(j)+'-'+str(i)+'.hdf5')
        ## Predictions
        train_pred = np.exp(model.predict_generator(generator=batch_generatorp(xtr,800,False), val_samples = xtr.shape[0])[:,0])-200
        print '*********************Training MAE over bag ',j,' and fold ',i,' is : ',mean_absolute_error(np.exp(ytr)-200,train_pred),'    ************************'
        pred = np.exp(model.predict_generator(generator=batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0])-200  ##Validation set predictions
        print '*********************Validation MAE over bag ',j,' and fold ',i,' is : ',mean_absolute_error(np.exp(yte)-200,pred),'    ************************'
        
        pred_test += np.exp(model.predict_generator(generator=batch_generatorp(xtest,800,False), val_samples = xtest.shape[0])[:,0]) - 200
        bag[inTe] = pred
        pred_oob[inTe] += pred
        i = i + 1
        
    print 'CV ',j,' took ',time.time()-start_time,' seconds'
    print 'Accuracy Over Bag ',j,' CV is : ',mean_absolute_error(bag,np.exp(y)-200)
    
pred_oob = pred_oob*1.0/(nbags*nfolds)
print 'Total MAE: ', mean_absolute_error(pred_oob,np.exp(y)-200)

##train Predictions
df = pd.DataFrame({'id':id_train, 'loss':pred_oob})
df.to_csv('pred_oob.csv',index=False)

##Test Predictions
pred_test = pred_test*1.0/(nbags*nfolds)
df = pd.DataFrame({'id':id_test, 'loss':pred_test})
df.to_csv('Submission Keras Shift Bag of CV.csv',index=False)
