#Author: Hencil Peter
#Date : 29/04/201
#calculateCost function calcualte teh squared error value for the given input values x, y and theta 

function[J] = calculateCost(x, y, theta)
 
 m = length(x);
 
 predictedValue = x * theta;
 
 squardErrorValue = (predictedValue - y ).^2;
 
 J = sum(squardErrorValue) / ( 2 * m );
 
 endfunction