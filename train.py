from utils import *

#Load Dataset
X_train, y_train = prepare_dataset(data_dir, 'train')
X_test, y_test = prepare_dataset(data_dir, 'test')
t = int(time.time())

#Normalizing
mean = np.mean(X_train,axis=(0,1,2,3))
std = np.std(X_train,axis=(0,1,2,3))
np.save('mean',mean)
np.save('std',std)
X_train = z_normalization(X_train, mean, std)
X_test = z_normalization(X_test, mean, std)

#Labels to binary
y_train_binary = keras.utils.to_categorical(y_train,num_classes)
y_test_binary = keras.utils.to_categorical(y_test,num_classes)

#CNN Model
model = create_CNN_model(X_train.shape[1:], num_classes, 0.3)
model, H = train_CNN_model(model, X_train, y_train_binary, X_test, y_test_binary, model_dir, t, batch_size=256, epochs=200)
