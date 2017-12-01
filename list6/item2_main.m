close all;
clear all;
clc;

% Load data
load ex6data2.mat;
% The load above loads X and y matrixes

% Plot data
plotData(X, y);

% Train
C = 1;
model1 = svmTrain(X, y, C, @sig1_gaussianKernel);
visualizeBoundary(X, y, model1);

model2 = svmTrain(X, y, C, @sig2_gaussianKernel);
visualizeBoundary(X, y, model2);
