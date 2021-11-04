import numpy as np


def calculate_mean(x_data):
    """
    This function uses numpy to calculate the mean vector of a dataset
    :param x_data: a 2-D numpy array with n_examples as rows and n_features as columns
    :return: mean_vec
    Note, you may not use np.mean to calculate the mean vector
    """
    cols = (len(x_data[0])) #get the number of columns
    mean_vector = np.array([]) #set up a new array to put our mean vector in
    for i in range(0, cols): #loop through the columns
        #concatenate the average of each column into the mean vector
        mean_vector = np.concatenate((mean_vector, sum(x_data[:, [i]])/len(x_data)), axis=None)
    return mean_vector



def calculate_cov(x_data, mean_vec):
    """
    This function uses numpy to calculate the covariance matrix of a dataset
    :param x_data: a 2-D numpy array with n_examples as rows and n_features as columns
    :return: mean_vec
    Note, you may not use np.cov to calculate the covariance matrix
    """
    #it's okay to use for loops for this, but it might be inefficient
    # (should still learn about inner product and outer product)
    #create empty covariance matrix
    cov_matrix = np.zeros(shape=((len(x_data[0])), (len(x_data[0]))))

    #get the number of columns
    cols = (len(x_data[0]))

    #get all the combinations of two columns using loops
    for col1 in range(cols):
        for col2 in range(cols):

            if col2 < col1:
                continue
            #print("columns we are looking at:", col1,col2)

            b = x_data[:, col1: col1+1]
            c = x_data[:, col2: col2+1]
            #print("column 1:", b, "column 2:", c)

            #get ave from mean_vec for these rows
            col1_mean = mean_vec[col1]
            col2_mean = mean_vec[col2]

            #get number of rows. for each row in these two columns: calculate the covariance
            covariance = 0
            for i in range(0, len(x_data)):
                #print("b[i]:", b[i])
                #print("c[i]:", c[i])
                #here is the actual covariance equation
                covariance += ((b[i] - col1_mean)*(c[i] - col2_mean))/(len(x_data)-1)

                #place covariance where it goes in matrix
                cov_matrix[col1, col2] = covariance
                cov_matrix[col2, col1] = covariance

    #print(cov_matrix)
    return cov_matrix

#def main():
 #   x_data = np.array([[1, 1, 1], [1, 2, 1], [1, 3, 2], [1, 4, 3]])
  #  vec = calculate_mean(x_data)
   # calculate_cov(x_data, vec)
#main()
