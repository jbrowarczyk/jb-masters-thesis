from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from utils import make_train_data, make_test_data, make_val_data, save_mlp, save_txt
from global_settings import TRAIN_VERBOSE
import numpy as np
import joblib
import os
import json

EXPERIMENT_NAME = "experiment4"

FEATURES = ["ar_16","ar_24","dwt","dwt_stat","welch_16","welch_32","welch_64"]

SAVE_RESULTS      = True  # saves results in single file using joblib library
SAVE_RESULTS_TXT  = True  # saves results in .txt file
SAVE_MODEL        = False # saves trained model

if TRAIN_VERBOSE:
	MLP_VERBOSE = 1
else:
	MLP_VERBOSE = 0

def experiment_mlp_singlelayer(train_data,train_data_classes,val_data,val_data_classes,test_data,test_data_classes,verbose):
	try:
		n_inputs = train_data.shape[1]

		encoder = LabelEncoder()
		train_data_classes_encoded = encoder.fit_transform(train_data_classes)
		test_data_classes_encoded = encoder.transform(test_data_classes)
		val_data_classes_encoded = encoder.transform(val_data_classes)

		train_data_classes_int = to_categorical(train_data_classes_encoded)
		test_data_classes_int = to_categorical(test_data_classes_encoded)
		val_data_classes_int = to_categorical(val_data_classes_encoded)

		model = Sequential()
		model.add(Dense(n_inputs,
			activation='relu',
			input_dim=n_inputs,
			kernel_initializer='he_uniform',
			bias_initializer='zeros'))
		model.add(Dense(3,
			activation='softmax',
			kernel_initializer='he_uniform',
			bias_initializer='zeros'))
		sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
		model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
		es = EarlyStopping(monitor='val_loss',
			mode='min',
			verbose=1,
			patience=50,
			restore_best_weights=True)
		history = model.fit(train_data,train_data_classes_int,batch_size=64,
			validation_data=(val_data,val_data_classes_int),
			callbacks=[es],epochs=2000,verbose=verbose)

		results = model.predict_classes(test_data,batch_size=128)
		results_decoded = encoder.inverse_transform(results)
		score = accuracy_score(test_data_classes_encoded,results)
		report = classification_report(test_data_classes,results_decoded,digits=4,output_dict=False)
		report_dict = classification_report(test_data_classes,results_decoded,output_dict=True)
		cm = confusion_matrix(test_data_classes,results_decoded)

		res = {}
		res['history'] = history
		res['results'] = results
		res['accuracy_score'] = score
		res['classification_report'] = report
		res['classification_report_dict'] = report_dict
		res['confusion_matrix'] = cm

		return res,model

	except Exception as e:
		print(e)
		return None,None


def experiment_mlp_1(train_data,train_data_classes,val_data,val_data_classes,test_data,test_data_classes,verbose):
	try:
		n_inputs = train_data.shape[1]

		encoder = LabelEncoder()
		train_data_classes_encoded = encoder.fit_transform(train_data_classes)
		test_data_classes_encoded = encoder.transform(test_data_classes)
		val_data_classes_encoded = encoder.transform(val_data_classes)

		train_data_classes_int = to_categorical(train_data_classes_encoded)
		test_data_classes_int = to_categorical(test_data_classes_encoded)
		val_data_classes_int = to_categorical(val_data_classes_encoded)

		model = Sequential()
		for j in range(3):
			model.add(Dense(n_inputs,
				activation='linear',
				input_dim=n_inputs,
				kernel_initializer='he_uniform',
				bias_initializer='zeros'))
			model.add(LeakyReLU(alpha=0.2,
				name='lrelu_0.2_1_' + str(j)))
		model.add(Dense(3,
			activation='softmax',
			kernel_initializer='he_uniform',
			bias_initializer='zeros'))
		sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
		model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
		es = EarlyStopping(monitor='val_loss',
			mode='min',
			verbose=1,
			patience=50,
			restore_best_weights=True)
		history = model.fit(train_data,train_data_classes_int,
			validation_data=(val_data,val_data_classes_int),
			callbacks=[es],epochs=2000,verbose=verbose)

		results = model.predict_classes(test_data,batch_size=128)
		results_decoded = encoder.inverse_transform(results)
		score = accuracy_score(test_data_classes_encoded,results)
		report = classification_report(test_data_classes,results_decoded,digits=4,output_dict=False)
		report_dict = classification_report(test_data_classes,results_decoded,output_dict=True)
		cm = confusion_matrix(test_data_classes,results_decoded)

		res = {}
		res['history'] = history
		res['results'] = results
		res['accuracy_score'] = score
		res['classification_report'] = report
		res['classification_report_dict'] = report_dict
		res['confusion_matrix'] = cm

		return res,model

	except Exception as e:
		print(e)
		return None,None

def experiment_mlp_2(train_data,train_data_classes,val_data,val_data_classes,test_data,test_data_classes,verbose):
	try:
		n_inputs = train_data.shape[1]

		encoder = LabelEncoder()
		train_data_classes_encoded = encoder.fit_transform(train_data_classes)
		test_data_classes_encoded = encoder.transform(test_data_classes)
		val_data_classes_encoded = encoder.transform(val_data_classes)

		train_data_classes_int = to_categorical(train_data_classes_encoded)
		test_data_classes_int = to_categorical(test_data_classes_encoded)
		val_data_classes_int = to_categorical(val_data_classes_encoded)

		model = Sequential()
		for j in range(4):
			model.add(Dense(n_inputs,
				activation='tanh',
				input_dim=n_inputs,
				kernel_initializer='he_uniform',
				bias_initializer='zeros'))
			model.add(LeakyReLU(alpha=0.2,
				name='lrelu_0.2_2_' + str(j)))
		model.add(Dense(3,
			activation='softmax',
			kernel_initializer='he_uniform',
			bias_initializer='zeros'))
		sgd = SGD(lr=0.005,decay=1e-6,momentum=0.9,nesterov=True)
		model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
		es = EarlyStopping(monitor='val_loss',
			mode='min',
			verbose=1,
			patience=250,
			restore_best_weights=True)
		history = model.fit(train_data,train_data_classes_int,
			validation_data=(val_data,val_data_classes_int),
			callbacks=[es],epochs=3000,verbose=verbose)

		results = model.predict_classes(test_data,batch_size=128)
		results_decoded = encoder.inverse_transform(results)
		score = accuracy_score(test_data_classes_encoded,results)
		report = classification_report(test_data_classes,results_decoded,digits=4,output_dict=False)
		report_dict = classification_report(test_data_classes,results_decoded,output_dict=True)
		cm = confusion_matrix(test_data_classes,results_decoded)

		res = {}
		res['history'] = history
		res['results'] = results
		res['accuracy_score'] = score
		res['classification_report'] = report
		res['classification_report_dict'] = report_dict
		res['confusion_matrix'] = cm

		return res,model

	except Exception as e:
		print(e)
		return None,None

def experiment_mlp_3(train_data,train_data_classes,val_data,val_data_classes,test_data,test_data_classes,verbose):
	try:
		n_inputs = train_data.shape[1]

		encoder = LabelEncoder()
		train_data_classes_encoded = encoder.fit_transform(train_data_classes)
		test_data_classes_encoded = encoder.transform(test_data_classes)
		val_data_classes_encoded = encoder.transform(val_data_classes)

		train_data_classes_int = to_categorical(train_data_classes_encoded)
		test_data_classes_int = to_categorical(test_data_classes_encoded)
		val_data_classes_int = to_categorical(val_data_classes_encoded)

		model = Sequential()
		for j in range(6):
			model.add(Dense(n_inputs,
				activation='relu',
				input_dim=n_inputs,
				kernel_initializer='he_uniform',
				bias_initializer='zeros'))
		model.add(Dense(3,
			activation='softmax',
			kernel_initializer='he_uniform',
			bias_initializer='zeros'))
		sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
		model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
		es = EarlyStopping(monitor='val_loss',
			mode='min',
			verbose=1,
			patience=70,
			restore_best_weights=True)
		history = model.fit(train_data,train_data_classes_int,
			validation_data=(val_data,val_data_classes_int),
			callbacks=[es],epochs=2000,verbose=verbose)

		results = model.predict_classes(test_data,batch_size=128)
		results_decoded = encoder.inverse_transform(results)
		score = accuracy_score(test_data_classes_encoded,results)
		report = classification_report(test_data_classes,results_decoded,digits=4,output_dict=False)
		report_dict = classification_report(test_data_classes,results_decoded,output_dict=True)
		cm = confusion_matrix(test_data_classes,results_decoded)

		res = {}
		res['history'] = history
		res['results'] = results
		res['accuracy_score'] = score
		res['classification_report'] = report
		res['classification_report_dict'] = report_dict
		res['confusion_matrix'] = cm

		return res,model

	except Exception as e:
		print(e)
		return None,None

def experiment_mlp_4(train_data,train_data_classes,val_data,val_data_classes,test_data,test_data_classes,verbose):
	try:
		n_inputs = train_data.shape[1]

		encoder = LabelEncoder()
		train_data_classes_encoded = encoder.fit_transform(train_data_classes)
		test_data_classes_encoded = encoder.transform(test_data_classes)
		val_data_classes_encoded = encoder.transform(val_data_classes)

		train_data_classes_int = to_categorical(train_data_classes_encoded)
		test_data_classes_int = to_categorical(test_data_classes_encoded)
		val_data_classes_int = to_categorical(val_data_classes_encoded)

		model = Sequential()
		for j in range(3):
			model.add(Dense(n_inputs,
				activation='tanh',
				input_dim=n_inputs,
				kernel_initializer='he_uniform',
				bias_initializer='zeros'))
		model.add(Dense(3,
			activation='softmax',
			kernel_initializer='he_uniform',
			bias_initializer='zeros'))
		sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
		model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
		es = EarlyStopping(monitor='val_loss',
			mode='min',
			verbose=1,
			patience=250,
			restore_best_weights=True)
		history = model.fit(train_data,train_data_classes_int,
			validation_data=(val_data,val_data_classes_int),
			callbacks=[es],epochs=3000,verbose=verbose)

		results = model.predict_classes(test_data,batch_size=128)
		results_decoded = encoder.inverse_transform(results)
		score = accuracy_score(test_data_classes_encoded,results)
		report = classification_report(test_data_classes,results_decoded,digits=4,output_dict=False)
		report_dict = classification_report(test_data_classes,results_decoded,output_dict=True)
		cm = confusion_matrix(test_data_classes,results_decoded)

		res = {}
		res['history'] = history
		res['results'] = results
		res['accuracy_score'] = score
		res['classification_report'] = report
		res['classification_report_dict'] = report_dict
		res['confusion_matrix'] = cm

		return res,model

	except Exception as e:
		print(e)
		return None,None

def main():
	if(EXPERIMENT_NAME not in os.listdir()):
		os.mkdir(EXPERIMENT_NAME)

	# neural networks with single hidden layer
	for feature in FEATURES:
		try:
			data = np.load(feature + "_stats.npy",allow_pickle=True).item()
			pca = joblib.load("pca_" + feature + "_stats_noval")

			train_data, train_data_classes = make_train_data(data,False)
			test_data, test_data_classes = make_test_data(data)
			val_data, val_data_classes = make_val_data(data)

			train_data_pca = np.array(pca.transform(train_data))
			test_data_pca = np.array(pca.transform(test_data))
			val_data_pca = np.array(pca.transform(val_data))

			train_data_classes = np.array(train_data_classes)
			val_data_classes = np.array(val_data_classes)
			test_data_classes = np.array(test_data_classes)

			print("Computing " + feature  + " single hidden layer neural network")
			res,model = experiment_mlp_singlelayer(train_data_pca,train_data_classes,val_data_pca,val_data_classes,test_data_pca,test_data_classes,MLP_VERBOSE)

			if res != None:
				if SAVE_RESULTS:
					filename = EXPERIMENT_NAME + "_" + feature + "_mlp_single_layer_results"
					path = os.path.join(EXPERIMENT_NAME,filename)
					joblib.dump(res,path)

				if SAVE_RESULTS_TXT:
					filename = EXPERIMENT_NAME + "_" + feature + "_mlp_single_layer_results.txt"
					path = os.path.join(EXPERIMENT_NAME,filename)
					save_txt(res,path)

				if SAVE_MODEL:
					filename = EXPERIMENT_NAME + "_" + feature + "_mlp_single_layer"
					path = os.path.join(EXPERIMENT_NAME,filename)
					save_mlp(model,path)

		except Exception as e:
			print("Error during " + EXPERIMENT_NAME + " " + feature  + " single hidden layer neural network")
			print(e)
			pass

	# neural networks with multiple hidden layers
	# welch32 only
	data = np.load("welch_32_stats.npy",allow_pickle=True).item()
	pca = joblib.load("pca_welch_32_stats_noval")

	train_data, train_data_classes = make_train_data(data,False)
	test_data, test_data_classes = make_test_data(data)
	val_data, val_data_classes = make_val_data(data)

	train_data_pca = np.array(pca.transform(train_data))
	test_data_pca = np.array(pca.transform(test_data))
	val_data_pca = np.array(pca.transform(val_data))

	train_data_classes = np.array(train_data_classes)
	val_data_classes = np.array(val_data_classes)
	test_data_classes = np.array(test_data_classes)

	print("Computing welch_32 multiple hidden layer neural network, specification 1:")
	print("3 hidden layers, LReLU activation (a = 0.02), learning rate = 0.01")
	print("decay = 1e-6, momentum = 0.9, patience = 50, max epochs = 2000")
	try:
		res,model = experiment_mlp_1(train_data_pca,train_data_classes,val_data_pca,val_data_classes,test_data_pca,test_data_classes,MLP_VERBOSE)
		if res != None:
			if SAVE_RESULTS:
				filename = EXPERIMENT_NAME + "_welch_32_mlp_multi_layer_spec_1_results"
				path = os.path.join(EXPERIMENT_NAME,filename)
				joblib.dump(res,path)

			if SAVE_RESULTS_TXT:
				filename = EXPERIMENT_NAME + "_welch_32_mlp_multi_layer_spec_1_results.txt"
				path = os.path.join(EXPERIMENT_NAME,filename)
				save_txt(res,path)

			if SAVE_MODEL:
				filename = EXPERIMENT_NAME + "_welch_32_mlp_multi_layer_spec_1"
				path = os.path.join(EXPERIMENT_NAME,filename)
				save_mlp(model,path)

	except Exception as e:
		print("Error during " + EXPERIMENT_NAME + " welch_32 multiple hidden layer neural network, specification 1")
		print(e)
		pass

	print("Computing welch_32 multiple hidden layer neural network, specification 2:")
	print("4 hidden layers, tanh + LReLU activation (a = 0.02), learning rate = 0.005")
	print("decay = 1e-6, momentum = 0.9, patience = 250, max epochs = 3000")
	try:
		res,model = experiment_mlp_2(train_data_pca,train_data_classes,val_data_pca,val_data_classes,test_data_pca,test_data_classes,MLP_VERBOSE)
		if res != None:
			if SAVE_RESULTS:
				filename = EXPERIMENT_NAME + "_welch_32_mlp_multi_layer_spec_2_results"
				path = os.path.join(EXPERIMENT_NAME,filename)
				joblib.dump(res,path)

			if SAVE_RESULTS_TXT:
				filename = EXPERIMENT_NAME + "_welch_32_mlp_multi_layer_spec_2_results.txt"
				path = os.path.join(EXPERIMENT_NAME,filename)
				save_txt(res,path)

			if SAVE_MODEL:
				filename = EXPERIMENT_NAME + "_welch_32_mlp_multi_layer_spec_2"
				path = os.path.join(EXPERIMENT_NAME,filename)
				save_mlp(model,path)

	except Exception as e:
		print("Some problem during " + EXPERIMENT_NAME + " welch_32 multiple hidden layer neural network, specification 2")
		print(e)
		pass

	print("Computing welch_32 multiple hidden layer neural network, specification 3:")
	print("6 hidden layers, ReLU activation, learning rate = 0.01")
	print("decay = 1e-6, momentum = 0.9, patience = 70, max epochs = 2000")
	try:
		res,model = experiment_mlp_3(train_data_pca,train_data_classes,val_data_pca,val_data_classes,test_data_pca,test_data_classes,MLP_VERBOSE)
		if res != None:
			if SAVE_RESULTS:
				filename = EXPERIMENT_NAME + "_welch_32_mlp_multi_layer_spec_3_results"
				path = os.path.join(EXPERIMENT_NAME,filename)
				joblib.dump(res,path)

			if SAVE_RESULTS_TXT:
				filename = EXPERIMENT_NAME + "_welch_32_mlp_multi_layer_spec_3_results.txt"
				path = os.path.join(EXPERIMENT_NAME,filename)
				save_txt(res,path)

			if SAVE_MODEL:
				filename = EXPERIMENT_NAME + "_welch_32_mlp_multi_layer_spec_3"
				path = os.path.join(EXPERIMENT_NAME,filename)
				save_mlp(model,path)

	except Exception as e:
		print("Error during " + EXPERIMENT_NAME + " welch_32 multiple hidden layer neural network, specification 3")
		print(e)
		pass

	print("Computing welch_32 multiple hidden layer neural network, specification 4:")
	print("3 hidden layers, tanh activation, learning rate = 0.01")
	print("decay = 1e-6, momentum = 0.9, patience = 250, max epochs = 3000")
	try:
		res,model = experiment_mlp_4(train_data_pca,train_data_classes,val_data_pca,val_data_classes,test_data_pca,test_data_classes,MLP_VERBOSE)
		if res != None:
			if SAVE_RESULTS:
				filename = EXPERIMENT_NAME + "_welch_32_mlp_multi_layer_spec_4_results"
				path = os.path.join(EXPERIMENT_NAME,filename)
				joblib.dump(res,path)

			if SAVE_RESULTS_TXT:
				filename = EXPERIMENT_NAME + "_welch_32_mlp_multi_layer_spec_4_results.txt"
				path = os.path.join(EXPERIMENT_NAME,filename)
				save_txt(res,path)

			if SAVE_MODEL:
				filename = EXPERIMENT_NAME + "_welch_32_mlp_multi_layer_spec_4"
				path = os.path.join(EXPERIMENT_NAME,filename)
				save_mlp(model,path)

	except Exception as e:
		print("Error during " + EXPERIMENT_NAME + " welch_32 multiple hidden layer neural network, specification 4")
		print(e)
		pass

if __name__ == "__main__":
	main()