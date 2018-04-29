## Author: Hencil Peter
## Created: 2018-04-29
# main file 


# load the test data set

dataset = load('TestDataSet.txt');
x = dataset(:, 1);
y = dataset(:, 2);

#apply mean normalization 
minX = min(x);
maxX = max(x);
meanNormalizedX = ( x - maxX ) / (maxX - minX);

#length of the dataset 
m = length(x);

#prepend x with ones 
normalizedXWithOnes = [ones(m, 1), meanNormalizedX];

#intial theta
theta = [1 1];

#iterations
iterations = 500

learningRate = 1;
[JValues, thetaValues] = gradientDescent(normalizedXWithOnes, y, theta', learningRate, iterations);







