## Author: Hencil Peter
## Date: 2018-04-29
# main file  - Linear Regression - Multivarient


# load the test data set
dataset = load('TestDataSetMultiVariate.txt');
sizeDataSet = size(dataset);

#extract only features list (i.e. except last column)
x = dataset(:, 1 : sizeDataSet(2)-1);

#extract the last column 
y = dataset(:, sizeDataSet(2));

#apply mean normalization 
#normalize x
for i = 1 : size(x, 2)
  minX = min(x(:,i));
  maxX = max(x(:, i));
  x(:, i) = (x(:, i) - maxX )  / (maxX - minX);
endfor
  
#normalize y  
minY = min(y);
maxY = max(y);
y = (y - maxY) / (maxY - minX);


#length of the dataset 
m = length(x);


#prepend x with ones 
normalizedXWithOnes = [ones(m, 1)  x];

n = size(normalizedXWithOnes, 2);

#intial theta
theta = ones(n, 1);

#iterations
iterations = 10

learningRate = 1;

#call gradient decent algorithm
[JValues, thetaValues] = calculateGradientMultiVariate(normalizedXWithOnes, y, theta', learningRate, iterations);








