from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


NUMERAL_SUFFIXES = ['-th', '-st', '-nd', '-rd', '-th']

def make_crossvalidation_data(split,x,y):
	train_data = []
	train_data_classes = []
	test_data = []
	test_data_classes = []

	for train_index in split[0]:
		train_data.append(x[train_index])
		train_data_classes.append(y[train_index])
	for test_index in split[1]:
		test_data.append(x[test_index])
		test_data_classes.append(y[test_index])

	return train_data, train_data_classes, test_data, test_data_classes

def numeral_str(numeral):
	num = abs(numeral)
	while num >= 10:
		num /= 10
	num = int(num)
	if num > 4:
		return str(numeral) + NUMERAL_SUFFIXES[4]
	else:
		return str(numeral) + NUMERAL_SUFFIXES[num]

def make_report(test_data_classes,results,digits=4):
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

	return res

def dots_to_underscores(s):
	return s.replace(".", "_")

