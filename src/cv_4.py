from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU
from keras.utils import to_categorical

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import numpy as np 
import os
import joblib
import random
import traceback

from utils import save_txt, save_mlp
from crossval_utils import make_crossvalidation_data, numeral_str, make_report, dots_to_underscores
from global_settings import TRAIN_VERBOSE

EXPERIMENT_NAME = "crossvalidation4"

FEATURES = ["welch_32"]

SAVE_SPLITS       = True 
SAVE_PCA          = False
SAVE_RESULTS      = False # saves results in single file using joblib library
SAVE_RESULTS_TXT  = True  # saves results in .txt file
SAVE_MODEL        = False # saves trained model
SAVE_ENCODER      = False

results_list = []
models_list  = []
pca_list     = []

def experiment_mlp_1a(train_data,train_data_classes,val_data,val_data_classes,test_data,test_data_classes,encoder,verbose):
	try:
		n_inputs = train_data.shape[1]

		if not encoder:
			encoder = LabelEncoder()

		train_data_classes_encoded = encoder.fit_transform(train_data_classes)
		test_data_classes_encoded  = encoder.transform(test_data_classes)
		val_data_classes_encoded   = encoder.transform(val_data_classes)

		train_data_classes_int = to_categorical(train_data_classes_encoded)
		test_data_classes_int  = to_categorical(test_data_classes_encoded)
		val_data_classes_int   = to_categorical(val_data_classes_encoded)

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
		res['results_decoded'] = results_decoded
		res['accuracy_score'] = score
		res['classification_report'] = report
		res['classification_report_dict'] = report_dict
		res['confusion_matrix'] = cm

		return res,model

	except Exception as e:
		print(e)
		return None,None

def main():
	for feature in FEATURES:
		try:
			print("Computing " + feature  + " multiple hidden layer neural network (specification 1) cross validation")
			data = np.load(feature + "_stats.npy",allow_pickle=True).item()

			meditation  = []
			music_video = []
			logic_game  = []

			for key in data.keys():
				for item in data[key]['meditation']:
					meditation.append(item)
				for item in data[key]['logic_game']:
					logic_game.append(item)
				for item in data[key]['music_video']:
					music_video.append(item)
		
			if len(meditation) != len(music_video) or len(music_video) != len(logic_game):
				raise Exception("Classes have unequal number of samples")
			len1 = len(meditation)

			x = meditation  + logic_game + music_video

			y = (len1*['meditation']) + (len1*['logic_game']) + (len1*['music_video'])
			if len(y) != len(x):
				raise Exception("Unequal number of input samples and output samples")

			kfold = KFold(n_splits = 10, shuffle = True)
			splits = list(kfold.split(x,y))

			encoder = LabelEncoder()
			encoder.fit(y)

			if EXPERIMENT_NAME not in os.listdir():
	 			os.mkdir(EXPERIMENT_NAME)

			if SAVE_SPLITS:
				np.save(os.path.join(EXPERIMENT_NAME,EXPERIMENT_NAME + "_" + feature + "_splits.npy"),splits)

			if SAVE_ENCODER:
				joblib.dump(encoder,os.path.join(EXPERIMENT_NAME,EXPERIMENT_NAME + "_" + feature + "_encoder"))

			predicted_list = np.ndarray(0)
			test_data_classes_list = np.ndarray(0)

			for i in range(len(splits)):
				print("Computing " + numeral_str(i+1) + " fold")

				initial_train_data, initial_train_data_classes, test_data, test_data_classes = make_crossvalidation_data(splits[i],x,y)

				l = len(initial_train_data)
				i1 = int(0.8889 * l)
				indices = [i for i in range(l)]

				random.shuffle(indices)
				train_val_indices = [indices[:i1], indices[i1:]]

				train_data, train_data_classes, val_data, val_data_classes = make_crossvalidation_data(train_val_indices,initial_train_data,initial_train_data_classes)

				if(SAVE_SPLITS):
					np.save(os.path.join(EXPERIMENT_NAME,EXPERIMENT_NAME + "_" + feature + "_test_val_splits_" + str(i) + ".npy"),train_val_indices)

				pca = PCA(n_components = 0.95)
				pca.fit(train_data)
				pca_list.append(pca)

				train_data_pca = np.array(pca.transform(train_data))
				val_data_pca   = np.array(pca.transform(val_data))
				test_data_pca  = np.array(pca.transform(test_data))

				res, model = experiment_mlp_1a(train_data_pca,train_data_classes,val_data_pca,val_data_classes,test_data_pca,test_data_classes,encoder,TRAIN_VERBOSE)
				results_list.append(res)
				models_list.append(model)

				test_data_classes_list = np.append(test_data_classes_list,test_data_classes)
				predicted_list = np.append(predicted_list,res['results_decoded'])

				if res != None:
					if SAVE_RESULTS:
						filename = EXPERIMENT_NAME + "_" + feature + "_mlp_multi_layer_spec_1_results_" + str(i+1)
						path = os.path.join(EXPERIMENT_NAME,filename)
						joblib.dump(res,path)

					if SAVE_RESULTS_TXT:
						filename = EXPERIMENT_NAME + "_" + feature + "_mlp_multi_layer_spec_1_results_" + str(i+1) + ".txt"
						path = os.path.join(EXPERIMENT_NAME,filename)
						save_txt(res,path)

					if SAVE_MODEL:
						filename = EXPERIMENT_NAME + "_" + feature + "_mlp_multi_layer_spec_1_" + str(i+1)
						path = os.path.join(EXPERIMENT_NAME,filename)
						save_mlp(model,path)

					if SAVE_PCA:
						filename = EXPERIMENT_NAME + "_" + feature + "_mlp_multi_layer_spec_1_pca_" + str(i+1)
						path = os.path.join(EXPERIMENT_NAME,filename)
						joblib.dump(pca,path)

			if SAVE_RESULTS or SAVE_RESULTS_TXT:
				average_report = make_report(test_data_classes_list,predicted_list)

			if SAVE_RESULTS:
				filename = EXPERIMENT_NAME + "_" + feature + "_mlp_multi_layer_spec_1_results_averaged"
				path = os.path.join(EXPERIMENT_NAME,filename)
				joblib.dump(average_report,path)

			if SAVE_RESULTS_TXT:
				filename = EXPERIMENT_NAME + "_" + feature + "_mlp_multi_layer_spec_1_results_averaged.txt"
				path = os.path.join(EXPERIMENT_NAME,filename)
				save_txt(average_report,path)

		except Exception as e:
			print("Error during " + EXPERIMENT_NAME + " " + feature  + "_mlp_multi_layer_spec_1")
			print(traceback.format_exc())
			pass

if __name__ == "__main__":
	main()