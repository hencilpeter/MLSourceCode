# Author : Hencil Peter
# Date : 30/04/2018
# Linear Regression using Single variable

import numpy as np
import csv

# read file content
def readFile(fileName):
    # open the file  in read only mode
    file = open("TestDataSetSingleVariable.txt", "r")

    # get the csv reader object
    csvReader = csv.reader(file, delimiter='\n')

    # empty file list
    fileList = []

    # read each line, parse it and append the same in the list
    for row in csvReader:
        data = row[0].split(',')
        fileList.append(data)

    # convert the list into integer array
    dataArray = np.array(fileList, dtype=np.float)

    # return the result back to caller
    return dataArray


# cost estimate function
def calculateCost(x, y, theta):
    # size of the data set
    m = len(x)

    # predicted Value
    hx = np.matmul(x, theta)

    # sequared Error
    squaredError = (hx - y).__pow__(2)

    # cost
    J = sum(squaredError) / (2 * m)

    return J


# calcualtes the gradient decent
def calculateGradient(x, y, theta, learningRate, iterations):
    # length of the input dataset
    m = len(x)

    # intialize the JValues with zeros
    JValues = np.zeros(iterations, dtype=np.float).reshape((iterations, 1))

    # initialize theta Values
    thetaValues = np.transpose(theta)

    for i in range(1, iterations):
        hx = np.matmul(x, thetaValues) - y

        # calcualte theta1 and theta2 values respectively
        thetaValues[0] = thetaValues[0] - (learningRate * 1 / m * np.matmul(hx.transpose(), x[:, 0]))
        thetaValues[1] = thetaValues[1] - (learningRate * 1 / m * np.matmul(hx.transpose(), x[:, 1]))

        # cost value (squared error value)
        JValues[i] = calculateCost(x, y, thetaValues)

        print("Iternation : ", i, "J(Theta) : ", thetaValues, "JValue : ", JValues[i])

    return thetaValues, JValues


# main function call
if __name__ == '__main__':
    # read the file content
    dataArray = readFile("TestDataSetSingleVariable")

    #size of the dataset
    arraySize = dataArray.shape
    m = arraySize[0]
    columns = arraySize[1]

    # extract/slicing x and y values
    x = np.array(dataArray[:, 0]).reshape((m, 1))
    y = np.array(dataArray[:, 1]).reshape((m, 1))

    # apply normalization
    minX = min(x)
    maxX = max(x)
    normalizedX = (x - maxX) / (maxX - minX)

    oneVector = np.ones(m, dtype=np.float).reshape((m, 1))
    normalizedXWithAdditionalColumn = np.append(oneVector, normalizedX, axis=1)


    # learning rate
    learningRate = 1

    # initialize theta values
    theta = np.ones(2, dtype=np.float).reshape((1, 2))

    iternations = 500

    #calcualte theta values for the given dataset
    newTheta, cost = calculateGradient(normalizedXWithAdditionalColumn, y, theta, learningRate, iternations)

