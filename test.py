from utils import *

X_test, y_test = prepare_dataset(data_dir, 'test')

mean = np.load('mean.npy')
std = np.load('std.npy')

X_test = z_normalization(X_test, mean, std)
y_test_binary = keras.utils.to_categorical(y_test,num_classes)

model = load_model(model_dir +'\\best_baseline_1')
cf_matrix, accuracy, macro_f1, mimsmatch = calculate_metrics(model, X_test, y_test_binary)
print('Accuracy : {}'.format(accuracy))
print('F1-score : {}'.format(macro_f1))

plot_confusion_matrix(cf_matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)