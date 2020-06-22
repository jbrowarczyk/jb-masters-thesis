from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from global_settings import TRAIN_VERBOSE
from utils import make_train_data, make_test_data, save_txt
import numpy as np
import joblib
import os


EXPERIMENT_NAME = "experiment3"

FEATURES = ["ar_16","ar_24","dwt","dwt_stat","welch_16","welch_32","welch_64"]
C_VALUES = [0.01,0.1,1,10,100]
GAMMA_VALUES = [0.1, 1, 10]

SKIP_COMBINATIONS = set([('dwt',10,0.1),('dwt',10,1),('dwt',10,10),
	                     ('dwt',100,0.1),('dwt',100,1),('dwt',100,10),
	                     ('dwt_stat',10,10),
	                     ('ar_24',100,0.1),('ar_24',100,1),('ar_24',100,10)])

SAVE_RESULTS      = True  # saves results in single file using joblib library
SAVE_RESULTS_TXT  = True  # saves results in .txt file
SAVE_MODEL        = False # saves trained model

def experiment_svm_rbf(train_data,train_data_classes,test_data,test_data_classes,c,gamma,verbose):
	try:
		svm = SVC(C=c,kernel='rbf',gamma=gamma,verbose=verbose)
		svm.fit(train_data,train_data_classes)

		results = svm.predict(test_data)
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

		return res,svm

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

			for c in C_VALUES:
				for gamma in GAMMA_VALUES:
					if (feature,c,gamma) in SKIP_COMBINATIONS:
						print("Skipping " + feature  + " SVM-rbf C = " + str(c) + " gamma = " + str(gamma))
						continue

					print("Computing " + feature  + " SVM-rbf C = " + str(c) + " gamma = " + str(gamma))
					res,model = experiment_svm_rbf(train_data_pca,train_data_classes,test_data_pca,test_data_classes,c,gamma,TRAIN_VERBOSE)

					if res != None:
						if SAVE_RESULTS:
							filename = EXPERIMENT_NAME + "_" + feature + " svm_rbf_c_" + str(c) + "_gamma_" + str(gamma) + "_results"
							path = os.path.join(EXPERIMENT_NAME,filename)
							joblib.dump(res,path)

						if SAVE_RESULTS_TXT:
							filename = EXPERIMENT_NAME + "_" + feature + " svm_rbf_c_" + str(c) + "_gamma_" + str(gamma) + "_results.txt"
							path = os.path.join(EXPERIMENT_NAME,filename)
							save_txt(res,path)

						if SAVE_MODEL:
							filename = EXPERIMENT_NAME + "_" + feature + " svm_rbf_c_" + str(c) + "_gamma_" + str(gamma) + "_model"
							path = os.path.join(EXPERIMENT_NAME,filename)
							joblib.dump(model,path)

		except Exception as e:
			print("Error during " + EXPERIMENT_NAME + " " + feature  + " SVM-RBF C = " + str(c) + " gamma = " + str(gamma))
			print(e)
			pass

if __name__ == "__main__":
	main()
