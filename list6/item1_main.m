close all;
clear all;
clc;

% Load data
load ex6data1.mat;
% The load above loads X and y matrixes

% Plot data
plotData(X, y);

% Traning
C = 1;
model1 = svmTrain(X, y, C, @linearKernel);
visualizeBoundaryLinear(X, y, model1);

C = 100;
model100 = svmTrain(X, y, C, @linearKernel);
visualizeBoundaryLinear(X, y, model100);

C = 0.001;
model0001 = svmTrain(X, y, C, @linearKernel);
visualizeBoundaryLinear(X, y, model0001);

y(37) = 1;
C = 1e9;
model1e9 = svmTrain(X, y, C, @linearKernel);
visualizeBoundaryLinear(X, y, model1e9);