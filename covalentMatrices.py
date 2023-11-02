import matplotlib.pyplot as plt
import numpy as np

# takes and image 28x28 and flattens it into shape (1,784)
def convertImage(image):
    ret = image.flatten()
    return ret

# finds the mean image to represent a number 0-9 in the training set by getting the sum of every equivalent pixel for all
# of that type of number and dividing by how many of that number is in the dataset
# I.E. there coudl be 1000 1's so I sum the (0,0), (0,1), (0,2), ..., (28,28) pixel for all 1000 1's and divide by 1000 to get a mean for each pixel
def meanMat(arr,width):
    mean = arr.sum(axis = 0)/width
    return mean

def covNum(images,labels,num):
    # I find every index of an inputted num 0-9 and make that the width of my array
    labelsIndex = np.where(labels == num)[0]
    width = len(labelsIndex)
    arr = np.empty((width,784))
    image = 0

    # I make an array width by 784 to hold all the flattened images of one type of number
    for i in labelsIndex:
        arr[image] = convertImage(images[i])
        image = image + 1

    # I find the mean image and subtract it from every image of a given number
    mean = meanMat(arr, width) 
    arr = arr - mean
    # Find the covariance matrix by taking array of all mean subtracted images and multiplying it by its transpose
    cov = np.dot(arr.transpose(), arr)

    #return the covariance matrix the array of all images to be used later for weight calculations and the mean image.
    return [cov, arr, mean]