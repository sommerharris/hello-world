from src.stats import calculate_mean
from src.stats import calculate_cov
import numpy as np
import math
import scipy
from scipy import linalg

class GaussianModel:
    def __init__(self, mean=None, cov=None):
        self.mean = mean
        self.cov = cov
        self.d = len(self.mean) # Set this to the feature dimension



    def calculate_log_likelihood(self, x):
        """
        Calculate the log-likelihood for
        :param x: example vector (containing feature values)
        :return: log likelihood
        """
        #get the determinant of the covariance matrix
        #det = np.linalg.det(self.cov)
        det = scipy.linalg.det(self.cov)
        print(det)

        #calculate the second expression and get the resulting 1x1 matrix
        second_expression = np.dot(np.dot(np.subtract(x, self.mean), np.linalg.inv(self.cov)), np.transpose((np.subtract(x, self.mean))))
        print(np.linalg.inv(self.cov))
        print(second_expression)
        #add log of first expression + second expression
        log_likelihood = -.5*(self.d * math.log(2*math.pi) + math.log(det) + second_expression)
        print(log_likelihood)

        return log_likelihood


    #This section is where I did my initial testing. The code started working today when I hard coded in covariance
    #matrix values that I knew would return a positive determinant
    #In general, this code would be much improved if I threw errors when the determinant is zero or negative.
#def main():

    #x_data = np.array([[3,-1], [-1, 7]])
    #vec = calculate_mean(x_data)
    #vec = np.mean(x_data, axis=0)
#    vec = np.array([0.55,-.25])
    #cov = calculate_cov(x_data, vec)
    #cov = np.cov(x_data, rowvar=False)
#    cov = np.array([[3,-1], [-1, 7]])
    #print(cov)
    #print(vec)
#    my_gaussian = GaussianModel(vec, cov)
#    test_data = np.array([4, -2])
#    print(my_gaussian.calculate_log_likelihood(test_data))
#main()
