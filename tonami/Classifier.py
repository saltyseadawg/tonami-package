#stub for classifier class
#could be 2 or 4 for num_classes
#method is method of classification. k-means, etc.
#k-fold stuff should also go in here, PCA, etc.

import pickle
import numpy as np
import numpy.typing as npt

class Classifier:
    def __init__(self, num_classes, method='svm'):
        self.num_classes = num_classes
        self.method = method

    def get_probability() -> npt.NDArray[float]:
        #associated probability of being in each class
        pass
    
    def classify_tones(self, features):
        # TODO: further development is required to use different classifier models, fit the data and classify the tone
        if self.method == 'svm':
            clf = pickle.load(open('tonami/data/pickled_svm_80.pkl', 'rb'))
            prediction = clf.predict(features)
        else:
            # other classifier models
            prediction = []
        return prediction
