close all;
clear all;
clc;

load ex1data2.txt;
x = ex1data2(:,[1 2]);
y = ex1data2(:,3);

% Item 1: run Stochastic Gradient Descendent
alpha = 0.01;
epochs = 100;
res = sgd_multi(x, y, alpha, epochs);

% Item 2.1: Final coeficients
disp("Weights [w0 ... wn]: "), disp(res.weights)

% Plot 2.2: Plot epochs by error
x_epochs = 1:1:epochs;
y_errors = res.errors;
figure (1);
plot(x_epochs, y_errors);

% Item 3: use least squares
batch_weights = batch(x, y);
disp("By batch weights [w0 ... wn]: "), disp(batch_weights)

% Item 3: Extra: computing error
x_size = size(x, 1);
x_batch = [ones(x_size, 1) x];
y_batch = x_batch * batch_weights;
error_batch = 0
for i = 1:1:x_size
  error = y(i) - y_batch(i);
  error_batch = error_batch + error * error;
endfor
error_batch = error_batch / x_size;
disp("Error in batch: "), disp(error_batch)
