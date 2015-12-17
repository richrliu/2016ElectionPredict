import cv2
import numpy as np

# 'Total population', 'Median age', '% BachelorsDeg or higher', 
# 'Unemployment rate', 'Per capita income', 'Total households', 
# 'Average household size', '% Owner occupied housing', '% Renter occupied housing', 
# '% Vacant housing', 'Median home value', 'Population growth', 
# 'House hold growth', 'Per capita income growth', 'Winner'

class StatModel(object):
    def load(self, filename):
        self.model.load(filename)
    def save(self, filename):
        self.model.save(filename)

class SVM(StatModel):
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):        self.model.train(samples, responses, 
            params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C = 1 ) )

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples])