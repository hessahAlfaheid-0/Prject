
import numpy as np
import pandas as pd
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import matplotlib.pyplot as plt
#import seaborn as sns



dataset = pd.read_csv('eeg-eye-state_csv.csv')

dataset.head()

X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:, -1].values-1

#normalize the data
Xmean = np.mean(X[:,:],axis=0)
Xstd = np.std(X[:,:],axis=0)

X = (X-Xmean)/Xstd


meta = [Xmean,Xstd]
meta = list(zip(*meta))
arr = np.array(meta)
df = pd.DataFrame()

df = pd.DataFrame(data=arr,columns=['Xmean','Xstd'])
df.to_csv('meta.csv',columns=['Xmean','Xstd'])
print(df)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)


import keras
from keras.models import Sequential  #use to initialize nnet
from keras.layers import  Dense      #use to cerat layesr in nnet
from keras.layers import  Dropout       


#initializing the ANN
model = Sequential()

model.add(Dense( units=50,kernel_initializer = 'uniform',activation='relu',input_shape=(14,)))
model.add(Dropout(p = 0.1))
model.add(Dense( units=40,kernel_initializer = 'uniform',activation='relu'))
model.add(Dropout(p = 0.1))
model.add(Dense( units=30,kernel_initializer = 'uniform',activation='sigmoid'))
model.add(Dropout(p = 0.1))
model.add(Dense( units=1,kernel_initializer = 'uniform',activation='sigmoid'))


#Fitting the ANN to the traning set
print ('Training...')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    X_train,
    y_train, 
    epochs=250, 
    batch_size=128,
#     verbose=0,
    validation_data=(X_test, y_test)
)
print ('Training Finished!')

train_loss, train_acc = model.evaluate(X_train,  y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=0)

print ('')
print ('Train accuracy:', train_acc)
print ('Test accuracy:', test_acc)

#%matplotlib inline

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('training&validation_accuracy.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('training&validation_loss.png')
plt.show()
plt.figure()

 
 
#matrix = metrics.confusion_matrix(y_test.argmax(axis=0), y_pred.argmax(axis=0))
 
from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(np.array(X_test))


labels = ['1', '2']
cm = confusion_matrix( y_test + 1,y_pred + 1)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.show()

model.save('model.h5')
#END OF Modeling

#---------------------------------------------------------
from keras.models import load_model
model_ = load_model('model.h5')

data_test = pd.read_csv('eeg-eye-state_csv.csv')
X = data_test.iloc[:,0:-1].values
y = data_test.iloc[:,-1].values


meta_ = pd.DataFrame()
meta_ = pd.read_csv('meta.csv')

Xmean =meta_['Xmean'].values
Xstd =meta_['Xstd'].values

idx = 572

X_  = ((X[idx])-Xmean)/Xstd
y_ = y[idx]

y_predict = model_.predict_classes(np.array([X_]))
y_predict = y_predict + 1

y_predict = y_predict.item((0, 0))
print ("Y = : {0}, Y_predict: {0:.2f}".format(y_,y_predict))

if y_ == y_predict and y_predict == 0:
  import smtplib

fromaddr = 'projectpsau.2020@gmail.com'
toaddrs  = 'hessahalfaheid1419@gmail.com'
msg = "\r\n".join([
  "From: user_me@gmail.com",
  "To: user_you@gmail.com",
  "Subject: Just a message",
  "",
  "islam adel"
  ])

# Credentials (if needed)
username = 'projectpsau.2020@gmail.com'
password =  '09870987H'
# The actual mail send
server = smtplib.SMTP('smtp.gmail.com:587')
server.ehlo()
server.starttls()
server.login(username,password)
server.sendmail(fromaddr, toaddrs, msg)
server.quit()
if y_ == y_predict and y_predict == 1:
  import serial
  ser = serial.Serial('/dev/ttyAMAO', 9600)
   ser.write('3')

