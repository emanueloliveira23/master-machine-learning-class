% Main entry point of list 2 - item 2 - Linear Regression with Regularization

clear all;
close all;
clc;


% Load
load "ex2data2.txt";
X = normalizeVector(ex2data2(:, [1, 2]));
Y = ex2data2(:, 3);

% Const
ALPHA = 0.01;
EPOCHS = 1000;
LAMBDAS = [0 0.01 0.25];
COUNT = size(X, 1);

% Plot
% item2_plot_data(X, Y);

% Map Features
complexX = mapFeature(X(:,1), X(:, 2));
complexX = [ones(COUNT, 1) complexX];

% Run
for lambda = LAMBDAS

  weights = regularStochasticGradDesc(
    complexX, Y,
    ALPHA, EPOCHS, lambda
  );
  
  % Ploting
  figure();
  item2_plot_data(X, Y);
  regularPlotDecisionBoundary(weights);
  
endfor