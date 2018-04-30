#Author : Hencil Peter
#Date : 29/04/201
#calculateGradient fucntion computes the theta values (for multivariate problem)

function [JValues, thetaValues] = calculateGradientMultiVariate (x, y, theta, learningRate, iterations)

#calculate the length
m = length(x);

#initialize  JValues and thetaValues
JValues = zeros(iterations, 1);

thetaValues = theta;
 
 #below loop repeats iterations time 
 for i = 1:iterations
   hx = (x * thetaValues' - y) ;
    
    for j = 1 : size(x,2)
      thetaValues(j) = thetaValues(j) - (learningRate * (hx' * x(:,j)))/m;
    endfor 
    
   JValues(i) = calculateCostMultiVariate(x, y, thetaValues');
   disp(sprintf("Iteration : %d , J(Theta) : %d, Theta : %d - %d", i, JValues(i), thetaValues));
   
 endfor

endfunction
