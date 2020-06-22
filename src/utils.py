import numpy as np
import matplotlib.pyplot as plt

def cm_norm(cm):
	return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

def make_train_data(data,use_val_as_train=False):
	train_data = []
	train_data_classes = []
	for sig_class in data['train']:
		for item in data['train'][sig_class]:
			train_data.append(item)
			train_data_classes.append(sig_class)
		if use_val_as_train:
			for item in data['val'][sig_class]:
				train_data.append(item)
				train_data_classes.append(sig_class)
	return train_data, train_data_classes

def make_test_data(data):
	test_data = []
	test_data_classes = []
	for sig_class in data['test']:
		for item in data['test'][sig_class]:
			test_data.append(item)
			test_data_classes.append(sig_class)
	return test_data, test_data_classes

def make_val_data(data):
	val_data = []
	val_data_classes = []
	for sig_class in data['val']:
		for item in data['val'][sig_class]:
			val_data.append(item)
			val_data_classes.append(sig_class)
	return val_data, val_data_classes


def plot_confusion_matrix(y_true,y_pred,classes=None,normalize=False,title=None,cmap=plt.cm.Blues):
    if not classes:
        classes = ['logic game','meditation','music video']
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm_norm(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=classes,
    	yticklabels=classes,title=title,ylabel='True class',
    	xlabel='Classifier output')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_confusion_matrix(cm,classes=None,normalize=False,title=None,cmap=plt.cm.Blues):
    if not classes:
        classes = ['logic game','meditation','music video']
    if normalize:
        cm = cm_norm(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=classes,
    	yticklabels=classes,title=title,ylabel='True class',
    	xlabel='Classifier output')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def save_mlp(model,path):
	json_model = model.to_json()
	filename_model = path + '_model'
	with open(filename_model,'w') as f:
		f.write(json_model)			
	filename_weights = path + '_weights.h5'
	model.save_weights(filename_weights, overwrite=True)

def save_txt(res,path):
	with open(path,'w') as f:
		f.write("accuracy: " + str(res['accuracy_score']))
		f.write("\n\n")
		f.write("classification report: \n")
		f.write(str(res['classification_report']))
		f.write("\n\n")
		f.write("confusion matrix: \n")
		f.write(str(res['confusion_matrix']))
		f.write("\n\n")
		f.write("normalised confusion matrix: \n")
		f.write(str(cm_norm(res['confusion_matrix'])))