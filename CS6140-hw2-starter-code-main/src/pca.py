from stats import calculate_mean, calculate_cov
import numpy as np


class PCA:
    def __init__(self, mean=None, cov=None):
        self.mean = None
        self.cov = None
        self.W = None

    def transform_matrix(self, k, x_data):
        """
        Given k and the covariance matrix
        return: W (transform matrix)
        """
        eig_val_cov, eig_vec_cov = np.linalg.eig(self.cov)

        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        W = ([])
        for i in range(len(eig_pairs)):
            print(i)
            W.append(eig_pairs[i][1])
            k = k - 1
            if k == 0:
                break

        # Visually confirm that the list is correctly sorted by decreasing eigenvalues
        #for i in eig_pairs:
            #print(i[0]) #these are the values
            #print(i[1])  # these are the vectors

        return W

    def calculate_pca(self, x_data, k):
        """
        Given x_data as the input data calculate
        a matrix to calculate a PCA matrix
        :return: W
        """
        self.mean = calculate_mean(x_data)
        self.cov = calculate_cov(x_data, self.mean)
        self.W = self.transform_matrix(k, x_data)

        return


def main():
    x_data = np.array([[1, 1, 3, 6], [1, 2, 3, 4], [1, 3, 3, 5]])
    vec = calculate_mean(x_data)
    calculate_cov(x_data, vec)
    my_pca = PCA()
    my_pca.calculate_pca(x_data, 2)
main()