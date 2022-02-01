#stub for classifier class
#could be 2 or 4 for num_classes
#method is method of classification. k-means, etc.
#k-fold stuff should also go in here, PCA, etc.

class Classifier:
    def __init__(self, num_classes, method):
        self.num_classes = num_classes
        self.method = method

    def get_probability() -> np.ndArray:
        #associated probability of being in each class
        pass
    