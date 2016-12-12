
# ## Using Keras Starter by danijelk

# In[20]:

import numpy as np
np.random.seed(123)
import pandas as pd
import subprocess
from scipy.sparse import csr_matrix,hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.callbacks import CSVLogger,EarlyStopping, ModelCheckpoint
from keras.regularizers import l2,l1,l1l2

# ## Batch Generator for training

# In[2]:

##Manually generate batches based on chenglong discussion

def batch_generator(X,Y,batch_size,shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:]
        Y_batch = Y[batch_index]
        counter += 1
        yield X_batch,Y_batch
        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


# ## Batch generator for testing purpose

# In[3]:

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0]/np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:]
        counter += 1
        yield X_batch
        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


# In[6]:

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')


# In[7]:

print 'Length of Train : ', len(train), ' Legth of Test : ',len(test)


# In[8]:

test['loss'] = np.nan
Y = train['loss'].as_matrix()
id_train = train['id'].as_matrix()
id_test = test['id'].as_matrix()


# In[9]:

ntrain = train.shape[0]
joined = pd.concat([train,test])


# ## categorical data

# In[10]:

f_cat = [f for f in joined.columns if 'cat' in f]    
tmp_cat = None
for f in f_cat:
    if tmp_cat is None:
        tmp_cat = pd.get_dummies(joined[f].astype('category'),prefix=f)
    tmp_cat = pd.concat([tmp_cat,pd.get_dummies(joined[f].astype('category'),prefix=f)],axis=1)

print tmp_cat.shape[0]


# ## Numerical Data

# In[11]:

f_num = [f for f in joined.columns if 'cont' in f]
scaler = StandardScaler()
tmp_num = pd.DataFrame(scaler.fit_transform(joined[f_num]),columns=f_num).set_index(joined.index)
print tmp_num.shape[0]
tmp = pd.concat([tmp_cat,tmp_num],axis=1)
tmp['loss'] = joined['loss']
del(tmp_cat,tmp_num)


# In[12]:

tmp.shape


# In[13]:

joined = tmp
train = joined[joined['loss'].notnull()]
test = joined[joined['loss'].isnull()]
print 'Dim Train : ',train.shape[0], 'Dim Test : ',test.shape[0]


# In[14]:

features = list(train.columns)
features.remove('loss')


# In[15]:

X_train = train[features].as_matrix()
Y_train = train['loss'].as_matrix()
##Shift more relevant for tree based regressors, not for DL models
#Shift = 200
#Y_train = np.log(Y_train+Shift)
X_test = test[features].as_matrix()


# In[16]:

del(joined,train,test,tmp)
##Remaining data points are X_train, Y_train, X_test


# ## Custome evaluation function for keras

# In[17]:

from keras import backend as K
def custom_mae(y_true,y_pred):
    return K.mean(K.abs(K.exp(y_pred) - K.exp(y_true)), axis=-1)

def custome_rmse(y_true,y_pred):
    return K.sqrt(K.mean(K.square(K.exp(y_pred) - K.exp(y_true)), axis=-1))

def custom_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


# ## Defining NN Architecture

# In[18]:

#import tensorflow as tf
#tf.python.control_flow_ops = tf
def nn_model():
    model = Sequential()
    model.add(Dense(260,W_regularizer = l2(),input_dim=X_train.shape[1],init='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(130,W_regularizer = l2(),init='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(65,W_regularizer = l2(),init='he_normal'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1,init='he_normal'))
    model.compile(loss='mae',optimizer='adam')
    return model
   


# In[19]:

model = nn_model()


# In[23]:

## CV returning indexes
nfolds = 5
folds = KFold(len(Y_train),n_folds=nfolds, shuffle=True,random_state=2016)


# In[32]:

## Training Model
i=0
nbags = 5
#nepochs = 60
pred_oob = np.zeros(X_train.shape[0])
pred_test = np.zeros(X_test.shape[0])
gModel = []

for (inTr, inTe) in folds:
    X = X_train[inTr]
    Y = Y_train[inTr]
    X_val = X_train[inTe]
    Y_val = Y_train[inTe]
    pred = np.zeros(X_val.shape[0])
    import time
    start_time = time.time()
    for j in range(nbags):
        #with open('log.txt','ab') as myfile:
        #    myfile.write('Beginning Training Fold '+str(i+1)+' bag '+str(j+1)+'\n')
            
        ##CSV Logger ... closes after all callbacks, thus local
        csv_logger = CSVLogger('log'+str(i)+'-'+str(j)+'.txt')
        model = nn_model()
        mtime = time.time() ##model time measure for each folds first bag        
        ## Using early stopping for automatic stopping
        earlyStopping=EarlyStopping(monitor='val_loss', patience=15, verbose=2, mode='min')
	checkpointer = ModelCheckpoint(filepath="models/Model"+str(i)+"-"+str(j)+".hdf5", verbose=1, save_best_only=True)
        fit = model.fit_generator(generator=batch_generator(X,Y,32,True),
                                  validation_data=(X_val,Y_val),
                                  callbacks=[csv_logger,earlyStopping,checkpointer],
                                  nb_epoch=500,
                                  samples_per_epoch=X.shape[0],
                                  verbose=1)
        gModel.append(model)
        model.save('models/Last_Iter'+str(i)+'_'+str(j)+'.h5')
	print '*********************Training Accuracy over fold ',j,' is : ',mean_absolute_error(Y,model.predict(X)[:,0]),'    ************************'
        #with open('log.txt','ab') as myfile:
        #    myfile.write('Done Training Fold '+str(i+1)+' bag '+str(j+1)+'\n')
        #if j==0:
        #    print '1st bag of ',i+1,' Fold took : ',time.time()-mtime,' seconds'
	#    break
        #print mean_absolute_error(np.exp(model.predict_generator(generator=batch_generatorp(X_val,800,False),val_samples=X_val.shape[0])[:,0])-Shift,np.exp(Y_val)-Shift)
        temp = model.predict_generator(generator=batch_generatorp(X_val,800,False),val_samples=X_val.shape[0])[:,0]
        #temp = np.exp(temp) - Shift
        pred += temp
        temp = model.predict_generator(generator=batch_generatorp(X_test,800,False),val_samples=X_test.shape[0])[:,0]
        #temp = np.exp(temp) - Shift
        pred_test += temp
    print i+1, ' Fold took ',time.time()-start_time,' seconds '
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(Y_val,pred)
    i += 1
    print 'Fold ',i,'-MAE:',score
    
print 'Total - MAE: ',mean_absolute_error(Y_train,pred_oob)

##train predictions
df = pd.DataFrame({'id':id_train,'loss':pred_oob})
df.to_csv('Pred_OOB_keras.csv',index=False)

##test predictions
pred_test /= nbags*nfolds
df = pd.DataFrame({'id':id_test,'loss':pred_test})
df.to_csv('submission_keras.csv',index=False)


