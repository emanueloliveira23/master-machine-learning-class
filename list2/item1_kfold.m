% Main entry point of list 2 - Linear Regression

clear all;
close all;
clc;


% Load
load "ex2data1.txt";
X = normalizeVector(ex2data1(:, [1, 2]));
Y = ex2data1(:, 3);

% Const
COUNT = size(X, 1);
ALL = 1:COUNT;
ALPHA = 0.01;
EPOCHS = 1000;
ATTRS = size(X, 2);

% K-fold
K = 5;
step = COUNT/K;
foldsStart = 1:step:COUNT;
foldsEnd = step:step:COUNT;
folds = [foldsStart(:) foldsEnd(:)];
weightSum = zeros(K, ATTRS + 1);

for k=1:K

  testStart = folds(k, 1);
  testEnd = folds(k, 2);

  % Test
  test = testStart:testEnd;
  xTest = X(test, :);
  yTest = Y(test, :);

  % Training
  training = setdiff(ALL, test);
  xTraining = X(training, :);
  yTraining = Y(training, :);

  % Run
  res = stochasticGradDesc(xTraining, yTraining, xTest, yTest, ALPHA, EPOCHS);
  weightSum(k, :) = res.weights(:);


endfor

% Computing result
weightsRows = size(weightSum, 1);
weightsCount = size(weightSum, 2);
weightsMean = zeros(weightsCount, 1);

for w=1:weightsCount
  weightsMean(w, 1) = sum(weightSum(:, w)) / weightsRows;
endfor

% Final Weights
disp("Pesos [w0 ... wn]: ");
disp(weightsMean);
