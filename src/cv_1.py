from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import numpy as np 
import os
import joblib
import traceback

from experiment1 import experiment_knn
from utils import save_txt
from crossval_utils import make_crossvalidation_data, numeral_str, dots_to_underscores

EXPERIMENT_NAME = "crossvalidation1"

COMBINATIONS = set([("welch_32",11),("dwt",17)])

SAVE_SPLITS       = True 
SAVE_PCA          = False
SAVE_RESULTS      = False # saves results in single file using joblib library
SAVE_RESULTS_TXT  = True  # saves results in .txt file
SAVE_MODEL        = False # saves trained model

results_list = []
models_list  = []
pca_list     = []

def main():
	for feature, k in COMBINATIONS:
		try:
			print("Computing " + feature  + " " + str(k) + "-NN 10-fold cross validation")
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

			x = meditation + logic_game + music_video

			y = (len1*['meditation']) + (len1*['logic_game']) + (len1*['music_video'])
			if len(y) != len(x):
				raise Exception("Unequal number of input samples and output samples")

			kfold = KFold(n_splits = 10, shuffle = True)
			splits = list(kfold.split(x,y))

			if EXPERIMENT_NAME not in os.listdir():
 				os.mkdir(EXPERIMENT_NAME)

			if SAVE_SPLITS:
				np.save(os.path.join(EXPERIMENT_NAME,EXPERIMENT_NAME + "_" + feature + "_splits.npy"),splits)

			predicted_list = np.ndarray(0)
			test_data_classes_list = np.ndarray(0)

			for i in range(len(splits)):
				print("Computing " + numeral_str(i+1) + " fold")

				train_data, train_data_classes, test_data, test_data_classes = make_crossvalidation_data(splits[i],x,y)

				pca = PCA(n_components = 0.95)
				pca.fit(train_data)
				pca_list.append(pca)
				train_data_pca = np.array(pca.transform(train_data))
				test_data_pca  = np.array(pca.transform(test_data))

				res, model = experiment_knn(train_data_pca,train_data_classes,test_data_pca,test_data_classes,k)
				results_list.append(res)
				models_list.append(model)

				test_data_classes_list = np.append(test_data_classes_list,test_data_classes)
				predicted_list = np.append(predicted_list,res['results'])

				if res != None:
					if SAVE_RESULTS:
						filename = EXPERIMENT_NAME + "_" + feature + "_" + str(k) + "nn_results_" + str(i+1)
						path = os.path.join(EXPERIMENT_NAME,filename)
						joblib.dump(res,path)

					if SAVE_RESULTS_TXT:
						filename = EXPERIMENT_NAME + "_" + feature + "_" + str(k) + "nn_results_" + str(i+1) + ".txt"
						path = os.path.join(EXPERIMENT_NAME,filename)
						save_txt(res,path)

					if SAVE_MODEL:
						filename = EXPERIMENT_NAME + "_" + feature + "_" + str(k) + "nn_model_" + str(i+1)
						path = os.path.join(EXPERIMENT_NAME,filename)
						joblib.dump(model,path)

					if SAVE_PCA:
						filename = EXPERIMENT_NAME + "_" + feature + "_" + str(k) + "-nn_pca_" + str(i+1)
						path = os.path.join(EXPERIMENT_NAME,filename)
						joblib.dump(pca,path)

			if SAVE_RESULTS or SAVE_RESULTS_TXT:
				average_report = make_report(test_data_classes_list,predicted_list)

				if SAVE_RESULTS:
					filename = EXPERIMENT_NAME + "_" + feature + "_" + str(k) + "nn_results_averaged"
					path = os.path.join(EXPERIMENT_NAME,filename)
					joblib.dump(average_report,path)

				if SAVE_RESULTS_TXT:
					filename = EXPERIMENT_NAME + "_" + feature + "_" + str(k) + "nn_results_averaged.txt"
					path = os.path.join(EXPERIMENT_NAME,filename)
					save_txt(average_report,path)

		except Exception as e:
			print("Error during " + EXPERIMENT_NAME + " " + feature + " " + str(k) + "-NN")
			print(traceback.format_exc())
			# print(e)
			pass

if __name__ == "__main__":
	main()
