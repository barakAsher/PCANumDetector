import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from numpy import linalg as LA
import covalentMatrices as CM


def eigenNum(cov):

    # find the eigenvvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = LA.eig(cov)

    # makes sure all eigenvector and eigenvalue values are real because eig in numpy sometimes returns imaginary numbers
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # sorting the eigenvalues and eigenvectors by the max eigenvalues to see which eigenvectors are most prominent for a given number
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    # find the total sum of eigenvalues and keep adding eigenvalues until I get 95% of the variance and use that index
    # to find the matrix of eigenvalues (linear independent features) that make up 95% of a given number

    #this step is used for data reduction because there are many eigenvectors that have such a small eigenvalue they don't make a difference in our guess
    value = sum(eigenvalues)
    e = 0
    index = 0
    currsum = 0
    while(e < .95):
        currsum = currsum + eigenvalues[index]
        e = currsum/value
        index = index + 1

    eigenvectors = eigenvectors[:,:index]

    # normalize the eigenvectors and multiply by 255 to get the value in grayscale for visualization purposes
    eigenvectors = 255*(eigenvectors)/(np.max(eigenvectors))

    # returns only the eigenvector array
    return eigenvectors

def minWeight(inputweights, numweights):
    """ params:

        inputweights: one dimensional flatten vector that is the target number to guess
        numweights: a matrix that represents the weights of eigenfaces that make up a given number
                    the matrix is how many of one type of number 0-9 there are in the training data by
                    784 which is the size of an image (28,28) flattened
    
    """

    # find the minimum euclidean distance between the weights of the inputted number and the weights of all numbers in the dataset 
    distances = np.sum(np.square(numweights-inputweights),axis=1)

    # return the minimum distance of the inputted number and its comparison to all 0's,1's,2's, ..., 9's
    return np.min(distances)

def weights(imagearray,labelsarray,testimagearray,testlabelarray):

    numWeights = []
    covMatrices = []
    meanMatrices = []
    eigNumbers = []

    for num in range(10):
        # for every number 0-9 find the covariance matrix, mean number image, eigenvectors, and weights
        cov = CM.covNum(imagearray,labelsarray,num)
        covMatrices.append(cov[0])
        eigNumbers.append(eigenNum(covMatrices[num]))
        meanMatrices.append(cov[2])
        numWeights.append(np.matmul(cov[1],eigNumbers[num]))

    count = 0
    n=1000

    for input in range(n):
        inputNum = input
        # get input num from test dataset flatten the input num from (28,28) to (784,)
        originalnumVec = CM.convertImage(testimagearray[inputNum])
        # get label to of test num to check if we are right
        num = testlabelarray[inputNum]
        # initalize a minIndex to keep track of what number is the best guess from 0-9
        # and keep max num to keep track of the smallest weight distance
        minIndex = 0
        maxNum = float('inf')
        for i in range(10):
            # subtract our flattened num by the mean matrix of a given number 0-9 and matrix multipy it with the eigenvectors of the same number
            # to get a weight vector to compare to the weights of other numbers in the dataset
            numVec = originalnumVec - meanMatrices[i]
            inputWeights = np.matmul(numVec, eigNumbers[i])
            temp = minWeight(inputWeights,numWeights[i])
            # if the euclidean distance between the weight vector of the input num and the guessed num 0-9 is smaller than the previous distance
            # update maxNum to be the new distance and the minIndex to be our new number guess
            if temp < maxNum:
                maxNum = temp
                minIndex = i
        if(num == minIndex): count = count + 1
        # output the correct answer and our guess for the first n numbers of the test dataset
        print(num,minIndex)
    # print our accuracy as a decimal
    print(count/1000)

def fileOpener():
    # read in the training and test files as numpy arrays

    trainimagefile = "Digits/t10k-images.idx3-ubyte"
    trainimagearray = idx2numpy.convert_from_file(trainimagefile)

    trainlabelFile = "Digits/t10k-labels.idx1-ubyte"
    trainlabelsarray = idx2numpy.convert_from_file(trainlabelFile)

    testimagefile = "Digits/train-images.idx3-ubyte"
    testimagearray = idx2numpy.convert_from_file(testimagefile)

    testlabelfile = "Digits/train-labels.idx1-ubyte"
    testlabelarray = idx2numpy.convert_from_file(testlabelfile)
    
    weights(trainimagearray, trainlabelsarray,testimagearray,testlabelarray)

fileOpener()