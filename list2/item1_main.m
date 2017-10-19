% Main entry point of list 2 - item 1 - Linear Regression

clear all;
close all;
clc;


% Load
load "ex2data1.txt";
X = ex2data1(:, [1, 2]);
Y = ex2data1(:, 3);
COUNT = size(X, 1);

% Plot
% source("item1_plot_data.m");

% Split
TRANING_SIZE = 70;
X_TRAINING = normalizeVector(X(1:TRANING_SIZE, :));
Y_TRAINING = Y(1:TRANING_SIZE, :);
X_TEST = normalizeVector(X(TRANING_SIZE+1:COUNT, :));
Y_TEST = Y(TRANING_SIZE+1:COUNT, :);

% Run
ALPHA = 0.01;
EPOCHS = 1000;
res = stochasticGradDesc(X_TRAINING, Y_TRAINING, X_TEST, Y_TEST, ALPHA, EPOCHS);

% Test
source("item1_test.m");