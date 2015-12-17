import csv
import cv2
import numpy as np
import classDefs

# Load serialized model
myModel = classDefs.SVM()
myModel.load('./potus_svm.xml')

with open('test_potus_by_county.csv') as csvfile:
	reader = csv.reader(csvfile)
	features = []
	rowNum = 0
	for row in reader:
		# Skip the first row with titles
		if rowNum != 0:
			rowVector = []
			for iString in range(0, len(row)):
				rowVector.append(float(row[iString]))
			features.append(rowVector)
		rowNum+=1
	# Convert features to NumPy array
	features = np.array(features, dtype = np.float32)

	# Zero mean unit variance normalize
	# features = (features-features.mean(axis=0))/features.std(axis=0)
	# for iCol in range(0, len(features[0])-1):
	# 	features[:, iCol] = (features[:,iCol] - features[:,iCol].mean(axis=0))/features[:,iCol].std(axis=0)

	# PCA on features
	mean, eigenvectors = cv2.PCACompute(features)
	testingFeats = np.array(np.matrix(features-mean)*np.matrix(eigenvectors))

	# Make Predictions
	myPredictions = myModel.predict(testingFeats)

	# Write predictions in file 
	outfile = open('predictions.csv', 'w')
	for p in myPredictions:
		if p:
			outfile.write('Mitt Romney\n')
		else:
			outfile.write('Barack Obama\n')