from global_settings import *
import numpy as np
from sklearn.decomposition import PCA
import scipy.signal
import scipy.stats
import spectrum
import math
import matplotlib.pyplot as plt
import pywt
import random
import joblib
import sys

def compute_autoregressive(eeg,order):
	output_structure = {}
	for subset_name in eeg.keys():
		output_structure.update({subset_name : {}})
		for class_name in eeg[subset_name].keys():
			output_structure[subset_name].update({class_name : []})
			for i in range(len(eeg[subset_name][class_name])):
				ar_models = []
				for j in range(eeg[subset_name][class_name][i].shape[1]):
					model = spectrum.arburg(eeg[subset_name][class_name][i][:,j],order=order,criteria=None)[0]
					model = [item.real for item in model]
					ar_models.append(model)
				output_structure[subset_name][class_name].append(np.array(ar_models).transpose())
	return output_structure

def compute_dwt(eeg,wavelet="db4",level=4):
	output_structure = {}
	for subset_name in eeg.keys():
		output_structure.update({subset_name : {}})
		for class_name in eeg[subset_name].keys():
			output_structure[subset_name].update({class_name : []})
			for i in range(len(eeg[subset_name][class_name])):
				transforms = []
				for j in range(eeg[subset_name][class_name][i].shape[1]):
					transform = pywt.wavedec(data=eeg[subset_name][class_name][i][:,j],wavelet=wavelet,level=level,axis=0)
					c = np.ndarray((0,))
					for l in range(len(transform)):
						c = np.append(c,transform[l])
					transforms.append(c)
				output_structure[subset_name][class_name].append(np.array(transforms).transpose())
	return output_structure

def compute_dwt_stat(eeg,wavelet="db4",level=4):
	output_structure = {}
	for subset_name in eeg.keys():
		output_structure.update({subset_name : {}})
		for class_name in eeg[subset_name].keys():
			output_structure[subset_name].update({class_name : []})
			for i in range(len(eeg[subset_name][class_name])):
				transforms = []
				for j in range(eeg[subset_name][class_name][i].shape[1]):
					transform = pywt.wavedec(data=eeg[subset_name][class_name][i][:,j],wavelet=wavelet,level=level,axis=0)
					c = []
					for item in transform:
						c.append(np.mean(item)) # np.mean(np.abs(transform))
						c.append(np.mean(np.abs(item)))
						c.append(np.var(item))
						c.append(scipy.stats.skew(item))
						c.append(scipy.stats.kurtosis(item))
						c.append(zero_crossings(item))
						c.append(sum(np.power(item,2)))
					transforms.append(np.array(c))
				output_structure[subset_name][class_name].append(np.array(transforms).transpose())
	return output_structure

def compute_pca(data,n_components,use_validation):
	training_data = []
	for sig_class in data['train'].keys():
		for item in data['train'][sig_class]:
			training_data.append(item)
	if use_validation:
		for sig_class in data['val'].keys():
			for item in data['val'][sig_class]:
				training_data.append(item)
	pca = PCA(n_components)
	pca.fit(training_data)
	return pca

def compute_welch(eeg,n_per_seg):
	output_structure = {}
	for subset_name in eeg.keys():
		output_structure.update({subset_name : {}})
		for class_name in eeg[subset_name].keys():
			output_structure[subset_name].update({class_name : []})
			for i in range(len(eeg[subset_name][class_name])):
				transformed_signals = scipy.signal.welch(eeg[subset_name][class_name][i],nperseg=n_per_seg,axis=0)[1]
				output_structure[subset_name][class_name].append(transformed_signals)
	return output_structure

def concatenate_channels(eeg):
	output_structure = {}
	for subset_name in eeg.keys():
		output_structure.update({subset_name : {}})
		for class_name in eeg[subset_name].keys():
			output_structure[subset_name].update({class_name : []})
			for frame in eeg[subset_name][class_name]:
				l = [frame[:,i] for i in range(frame.shape[1])]
				newframe = np.concatenate(l)
				output_structure[subset_name][class_name].append(newframe)
	return output_structure

def concatenate_feature_vectors(eeg,stats):
	if eeg.keys() != stats.keys():
		raise ValueError("Input dictionaries have different keys")
	output_structure = {}
	for subset_name in eeg.keys():
		if eeg[subset_name].keys() != stats[subset_name].keys():
			raise ValueError("Input dictionaries have different keys")
		output_structure.update({subset_name : {}})
		for class_name in eeg[subset_name].keys():
			if len(eeg[subset_name][class_name]) != len(stats[subset_name][class_name]):
				raise ValueError("Numbers of frames in dicts are different")
			output_structure[subset_name].update({class_name : []})
			for i in range(len(eeg[subset_name][class_name])):
				frame1 = eeg[subset_name][class_name][i] 
				frame2 = stats[subset_name][class_name][i] 
				if len(frame1.shape) != 1 or len(frame2.shape) != 1:
					raise ValueError("Frame dimensionality not equal to 1")
				newframe = np.concatenate([frame1, frame2])
				output_structure[subset_name][class_name].append(newframe)
	return output_structure

def zero_crossings(data):
	return ((data[:-1] * data[1:]) < 0).sum()

def main():
	try:
		print("Loading preprocessed data...")
		ica_data = np.load("preprocessed_data.npy",allow_pickle = True).item()
		stats = np.load("stats.npy",allow_pickle = True).item()
		stats_concat = concatenate_channels(stats)
		print("Preprocessed data loaded")

	except Exception as e:
		print("Error during loading preprocessed data")
		print(e)
		sys.exit()

	try:
		print("Computing ar_16 features...") 
		ar_16 = compute_autoregressive(ica_data,16)
		print("ar_16 features computed")
		#np.save("ar_16.npy",ar_16)

		print("Concatenating ar_16 features with means and variances...")
		ar_16_concat = concatenate_channels(ar_16)
		ar_16_stats = concatenate_feature_vectors(ar_16_concat,stats_concat)
		print("ar_16 features concatenated")

		print("Saving ar_16 features...")
		np.save("ar_16_stats.npy",ar_16_stats)
		print("ar_16 features saved")

		print("Computing PCA on ar_16 features...")
		pca_ar_16_stats = compute_pca(ar_16_stats,0.95,True)
		pca_ar_16_stats_no_validation = compute_pca(ar_16_stats,0.95,False)
		print("ar_16 PCA computed")

		print("Saving ar_16 PCA...")
		joblib.dump(pca_ar_16_stats,"pca_ar_16_stats")
		joblib.dump(pca_ar_16_stats_no_validation,"pca_ar_16_stats_noval")
		print("ar_16 PCA saved")

		print("ar_16 done")

	except Exception as e:
		print("Error during computing ar_16 features")
		print(e)
		pass

	try:
		print("Computing ar_24 features...") 
		ar_24 = compute_autoregressive(ica_data,24)
		print("ar_24 features computed")
		#np.save("ar_24.npy",ar_24)

		print("Concatenating ar_24 features with means and variances...")
		ar_24_concat = concatenate_channels(ar_24)
		ar_24_stats = concatenate_feature_vectors(ar_24_concat,stats_concat)
		print("ar_24 features concatenated")

		print("Saving ar_24 features...")
		np.save("ar_24_stats.npy",ar_24_stats)
		print("ar_24 features saved")

		print("Computing PCA on ar_24 features...")
		pca_ar_24_stats = compute_pca(ar_24_stats,0.95,True)
		pca_ar_24_stats_no_validation = compute_pca(ar_24_stats,0.95,False)
		print("ar_24 PCA computed")

		print("Saving ar_24 PCA...")
		joblib.dump(pca_ar_24_stats,"pca_ar_24_stats")
		joblib.dump(pca_ar_24_stats_no_validation,"pca_ar_24_stats_noval")
		print("ar_24 PCA saved")

		print("ar_24 done")

	except Exception as e:
		print("Error during computing ar_24 features")
		print(e)
		pass

	try:
		print("Computing welch_16 features...") 
		welch_16 = compute_welch(ica_data,16)
		print("welch_16 features computed")
		#np.save("welch_16.npy",welch_16)

		print("Concatenating welch_16 features with means and variances...")
		welch_16_concat = concatenate_channels(welch_16)
		welch_16_stats = concatenate_feature_vectors(welch_16_concat,stats_concat)
		print("welch_16 features concatenated")

		print("Saving welch_16 features...")
		np.save("welch_16_stats.npy",welch_16_stats)
		print("welch_16 features saved")

		print("Computing PCA on welch_16 features...")
		pca_welch_16_stats = compute_pca(welch_16_stats,0.95,True)
		pca_welch_16_stats_no_validation = compute_pca(welch_16_stats,0.95,False)
		print("welch_16 PCA computed")

		print("Saving welch_16 PCA...")
		joblib.dump(pca_welch_16_stats,"pca_welch_16_stats")
		joblib.dump(pca_welch_16_stats_no_validation,"pca_welch_16_stats_noval")
		print("welch_16 PCA saved")

		print("welch_16 done")

	except Exception as e:
		print("Error during computing welch_16 features")
		print(e)
		pass

	try:
		print("Computing welch_32 features...") 
		welch_32 = compute_welch(ica_data,32)
		print("welch_32 features computed")
		#np.save("welch_32.npy",welch_32)

		print("Concatenating welch_32 features with means and variances...")
		welch_32_concat = concatenate_channels(welch_32)
		welch_32_stats = concatenate_feature_vectors(welch_32_concat,stats_concat)
		print("welch_32 features concatenated")

		print("Saving welch_32 features...")
		np.save("welch_32_stats.npy",welch_32_stats)
		print("welch_32 features saved")

		print("Computing PCA on welch_32 features...")
		pca_welch_32_stats = compute_pca(welch_32_stats,0.95,True)
		pca_welch_32_stats_no_validation = compute_pca(welch_32_stats,0.95,False)
		print("welch_32 PCA computed")

		print("Saving welch_32 PCA...")
		joblib.dump(pca_welch_32_stats,"pca_welch_32_stats")
		joblib.dump(pca_welch_32_stats_no_validation,"pca_welch_32_stats_noval")
		print("welch_32 PCA saved")

		print("welch_32 done")

	except Exception as e:
		print("Error during computing welch_32 features")
		print(e)
		pass

	try:
		print("Computing welch_64 features...") 
		welch_64 = compute_welch(ica_data,64)
		print("welch_64 features computed")
		#np.save("welch_64.npy",welch_64)

		print("Concatenating welch_64 features with means and variances...")
		welch_64_concat = concatenate_channels(welch_64)
		welch_64_stats = concatenate_feature_vectors(welch_64_concat,stats_concat)
		print("welch_64 features concatenated")

		print("Saving welch_64 features...")
		np.save("welch_64_stats.npy",welch_64_stats)
		print("welch_64 features saved")

		print("Computing PCA on welch_64 features...")
		pca_welch_64_stats = compute_pca(welch_64_stats,0.95,True)
		pca_welch_64_stats_no_validation = compute_pca(welch_64_stats,0.95,False)
		print("welch_64 PCA computed")

		print("Saving welch_64 PCA...")
		joblib.dump(pca_welch_64_stats,"pca_welch_64_stats")
		joblib.dump(pca_welch_64_stats_no_validation,"pca_welch_64_stats_noval")
		print("welch_64 PCA saved")

		print("welch_64 done")

	except Exception as e:
		print("Error during computing welch_64 fatures")
		print(e)
		pass

	try:
		print("Computing dwt features...") 
		dwt = compute_dwt(ica_data)
		print("dwt features computed")
		#np.save("dwt.npy",dwt)

		print("Concatenating dwt features with means and variances...")
		dwt_concat = concatenate_channels(dwt)
		dwt_stats = concatenate_feature_vectors(dwt_concat,stats_concat)
		print("dwt features concatenated")

		print("Saving dwt features...")
		np.save("dwt_stats.npy",dwt_stats)
		print("dwt features saved")

		print("Computing PCA on dwt features...")
		pca_dwt_stats = compute_pca(dwt_stats,0.95,True)
		pca_dwt_stats_no_validation = compute_pca(dwt_stats,0.95,False)
		print("dwt PCA computed")

		print("Saving dwt PCA...")
		joblib.dump(dwt_stats,"pca_dwt_stats")
		joblib.dump(pca_dwt_stats_no_validation,"pca_dwt_stats_noval")
		print("dwt PCA saved")

		print("dwt done")

	except Exception as e:
		print("Error during computing dwt features")
		print(e)
		pass

	try:
		print("Computing dwt_stat features...")
		dwt_stat = compute_dwt_stat(ica_data)
		print("dwt_stat features computed")
		#np.save("dwt_stat.npy",dwt_stat)

		print("Concatenating dwt_stat features with means and variances...")
		dwt_stat_concat = concatenate_channels(dwt_stat)
		dwt_stat_stats = concatenate_feature_vectors(dwt_stat_concat,stats_concat)
		print("dwt_stat features concatenated")

		print("Saving dwt_stat features...")
		np.save("dwt_stat_stats.npy",dwt_stat_stats)
		print("dwt_stat features saved")

		print("Computing PCA on dwt_stat features...")
		pca_dwt_stat_stats = compute_pca(dwt_stat_stats,0.95,True)
		pca_dwt_stat_stats_no_validation = compute_pca(dwt_stat_stats,0.95,False)
		print("dwt_stat PCA computed")

		print("Saving dwt_stat PCA...")
		joblib.dump(dwt_stat_stats,"pca_dwt_stat_stats")
		joblib.dump(pca_dwt_stat_stats_no_validation,"pca_dwt_stat_stats_noval")
		print("dwt_stat PCA saved")

		print("dwt_stat done")

	except Exception as e:
		print("Error during computing dwt_stat features")
		print(e)
		pass

if __name__ == "__main__":
	main()