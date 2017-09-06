close all;
clear all;
clc;

load ex1data1.txt;
x = ex1data1(:,1);
y = ex1data1(:,2);

% Item 1: plot data
% plot(x, y, '+');

% Item 2: run Stochastic Gradient Descendent
source("sgd.m");
alpha = 0.001;
epochs = 1000;
res = sgd_multi(x, y, alpha, epochs);

% Item 2.1: Final coeficients
disp("Weights [w0 ... wn]: "), disp(res.weights)

% Plot 2.2: Plot epochs by error
x_epochs = 1:1:epochs;
y_errors = res.errors;
figure (1);
plot(x_epochs, y_errors);

% Extra: plot linear function compared to data
size_x = max(x);
x_test = 1:1:size_x;
y_test = zeros(size_x, 1);
for i = 1:size_x 
  y_test(i, 1) = [1 i] * res.weights;
endfor
figure (2);
plot(x,y,'+b',x_test,y_test,'r');