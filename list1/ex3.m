clear all;
close all;
clc;

load ex1data3.txt
x = ex1data3(:, 1:5);
y = ex1data3(:, 6);

training_size = 31;
test_size = size(x, 1) - training_size;

training_x = x(1:training_size, :);
training_y = y(1:training_size, :);

test_x = x(training_size+1:training_size+test_size, :);
test_y = x(training_size+1:training_size+test_size, :);

