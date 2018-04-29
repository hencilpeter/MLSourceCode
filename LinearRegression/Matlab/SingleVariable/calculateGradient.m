#Author : Hencil Peter
#Date : 29/04/201
#calculateGradient fucntion compute the theta values (for univarient problem)
#
function [JValues, thetaValues] = calculateGradient (x, y, theta, learningRate, iterations)

#calculate the length
m = length(x);

#initialize  JValues and thetaValues
JValues = zeros(iterations, 1);

thetaValues = theta;
 
 #below loop repeats iterations time 
 for i = 1:iterations
   hx = (x * thetaValues - y) ;
   
   thetaValues(1) = thetaValues(1) - (learningRate * (hx' * x(:,1)))/m;
   thetaValues(2) = thetaValues(2) - (learningRate * (hx' * x(:,2)))/m;
   
   #below two lines also correct
   #thetaValues(1) = thetaValues(1) - learningRate * 1/m * hx * x(:,1);
   #thetaValues(2) = thetaValues(2) - learningRate * 1/m * hx * x(:,2);
   
   JValues(i) = calculateCost(x, y, thetaValues);
   
   disp(sprintf("Iteration : %d , J(Theta) : %d, Theta : %d - %d", i, JValues(i), thetaValues));
 endfor

endfunction
