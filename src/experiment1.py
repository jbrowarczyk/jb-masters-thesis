from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import joblib
import os
from utils import make_train_data, make_test_data, save_txt

EXPERIMENT_NAME = "experiment1"

FEATURES = ["ar_16","ar_24","dwt","dwt_stat","welch_16","welch_32","welch_64"]
K_VALUES = [5,7,11,14,17]

SKIP_COMBINATIONS = set([])

SAVE_RESULTS      = True  # saves results in single file using joblib library
SAVE_RESULTS_TXT  = True  # saves results in .txt file
SAVE_MODEL        = False # saves trained model

def experiment_knn(train_data,train_data_classes,test_data,test_data_classes,k):
	try:
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(train_data,train_data_classes)

		results = knn.predict(test_data)
		score = accuracy_score(test_data_classes,results)
		report = classification_report(test_data_classes,results,digits=4,output_dict=False)
		report_dict = classification_report(test_data_classes,results,output_dict=True)
		cm = confusion_matrix(test_data_classes,results)

		res = {}
		res['results'] = results
		res['accuracy_score'] = score
		res['classification_report'] = report
		res['classification_report_dict'] = report_dict
		res['confusion_matrix'] = cm

		return res,knn

	except Exception as e:
		print(e)
		return None,None

def main():
	if(EXPERIMENT_NAME not in os.listdir()):
		os.mkdir(EXPERIMENT_NAME)

	for feature in FEATURES:
		try:
			data = np.load(feature + "_stats.npy",allow_pickle=True).item()
			pca = joblib.load("pca_" + feature + "_stats")
			train_data, train_data_classes = make_train_data(data,True)
			test_data, test_data_classes = make_test_data(data)
			train_data_pca = np.array(pca.transform(train_data))
			test_data_pca = np.array(pca.transform(test_data))

			for k in K_VALUES:
				if (feature,k) in SKIP_COMBINATIONS:
					print("Skipping " + feature  + " " + str(k) + "-NN")
					continue

				print("Computing " + feature  + " " + str(k) + "-NN")
				res, model = experiment_knn(train_data_pca,train_data_classes,test_data_pca,test_data_classes,k)

				if res != None:
					if SAVE_RESULTS:
						filename = EXPERIMENT_NAME + "_" + feature + "_" + str(k) + "nn_results"
						path = os.path.join(EXPERIMENT_NAME,filename)
						joblib.dump(res,path)

					if SAVE_RESULTS_TXT:
						filename = EXPERIMENT_NAME + "_" + feature + "_" + str(k) + "nn_results.txt"
						path = os.path.join(EXPERIMENT_NAME,filename)
						save_txt(res,path)

					if SAVE_MODEL:
						filename = EXPERIMENT_NAME + "_" + feature + "_" + str(k) + "nn_model"
						path = os.path.join(EXPERIMENT_NAME,filename)
						joblib.dump(model,path)

		except Exception as e:
			print("Error during " + EXPERIMENT_NAME + " " + feature + " " + str(k) + "-NN")
			print(e)
			pass

if __name__ == "__main__":
	main()
