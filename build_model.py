import csv
import cv2
import numpy as np
import classDefs   # Wrapper class for OpenCV StatModel and Support Vector Machine

with open('train_potus_by_county.csv') as csvfile:
	reader = csv.reader(csvfile)
	features = []
	labels = []
	rowNum = 0
	for row in reader:
		if rowNum != 0:  #skip the first line with the titles
			rowVector = []  
			# Build list of features
			for iString in range(0, len(row)-1): 
				rowVector.append(float(row[iString]))
			# Process labels of observations
			if row[len(row)-1] == 'Mitt Romney': 
				labels.append(float(1))
			else:
				labels.append(float(0))
			features.append(rowVector)
		rowNum+=1
	#Convert to NumPy array for OpenCV compatibility
	features = np.array(features, dtype = np.float32)
	labels = np.array(labels, dtype = np.float32)

	# Zero Mean Unit Variance Normalization
	# features = (features-features.mean(axis=0))/features.std(axis=0)
	# for iCol in range(0, len(features[0])-1):
	# 	features[:, iCol] = (features[:,iCol] - features[:,iCol].mean(axis=0))/features[:,iCol].std(axis=0)

	# Principal Component Analysis 
	# Project features onto uncorrelated component space
	mean, eigenvectors = cv2.PCACompute(features)
	trainingFeats = np.array(np.matrix(features-mean)*np.matrix(eigenvectors))

	# Train SVM
	mySvm = classDefs.SVM()
	mySvm.train(trainingFeats, labels)
	mySvm.save('./potus_svm.xml')

	# Gauge performance of this model -> performance.txt
	resp = mySvm.predict(trainingFeats)
	err = (labels!=resp).mean()
	# print 'error: %.2f %%' % (err*100)

	# Write comments to file
	outfile = open('performance.txt', 'w')
	outfile.write('** Model: Linear SVM with parameters defined in potus_svm.xml\n')
	outfile.write('** Trained on: 14 features, 1213 observations in train_potus_by_county.csv \n')
	outfile.write('** Results for testing model on the training data: \n')
	outfile.write('  Error: %.2f %% \n' % (err*100))
	outfile.write('  (We expect that the error rate in the testing data >= %.2f %%)\n\n' % (err*100))
	outfile.write('Richard Liu, Nov 19 2015')
