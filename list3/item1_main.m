close all;
clear all;
clc;

% Load


load ex3data1.mat;

% load X matrix
% load T matrix

[xRowsCount xColsCount] = size(X);
[yRowsCount yColsCount] = size(T);

trainingSetSize = 4000;
validationSetSize = 500;
testSetSize = 500;

trainingSetStart = 1;
trainingSetEnd = trainingSetSize;
validationSetStart = trainingSetEnd + 1;
validationSetEnd = validationSetStart + validationSetSize - 1;
testSetStart = validationSetEnd + 1;
testSetEnd = testSetStart + testSetSize - 1;

alpha = 0.0001;
hiddenLayerNeuronsCount = 10;
outputLayerNeuronsCount = 10;

inputToHiddenWeights = 0.2 * rand(hiddenLayerNeuronsCount, xColsCount + 1); % + 1 >> w0
hiddenToOutputWeights = 0.2 * rand(outputLayerNeuronsCount, hiddenLayerNeuronsCount + 1); % + 1 >> w0


% Traning

lastValidationError = inf;
trainingErrorEpoch = [];
validationErrorEpoch = [];

growthControl = 2; 

while growthControl > 0
	

	indexes = randperm(xRowsCount)';

	xTraningSet = X(indexes(trainingSetStart:trainingSetEnd, :), :);
	yTraningSet = T(indexes(trainingSetStart:trainingSetEnd,:), :);

	xValidationSet = X(indexes(validationSetStart:validationSetEnd), :);
	yValidationSet = T(indexes(validationSetStart:validationSetEnd), :);


	% Traning

	errors = zeros(trainingSetSize, yColsCount);

	for i = 1:trainingSetSize

		% Input >> Hidden
		Xi = [-1 xTraningSet(i,:)]';
		Ui = inputToHiddenWeights * Xi;
		Yi = 1./(1+exp(-Ui));

		% Hidden >> Output
		Zi = [-1; Yi];
		Vi = hiddenToOutputWeights * Zi;
		Oi = 1./(1+exp(-Vi));

		% Error
		Ei = yTraningSet(i) - Oi;

		% Local Gradients
		sigk = Oi.*(1 - Oi);
		sigk = Ei.*sigk;

		sigi = Yi.*(1 - Yi);
		sigi = sigi.*(hiddenToOutputWeights(:,2:end)'*sigk);

		% Update weights
		hiddenToOutputWeights = hiddenToOutputWeights + alpha*sigk*Zi';
		inputToHiddenWeights = inputToHiddenWeights + alpha*sigi*Xi';

		errors(i, :) = Ei;

	endfor % traning sample

	trainingErrorEpoch = [
		trainingErrorEpoch; 
		sum(sum(errors.^2)) / trainingSetSize
	];


	% Validation

	errors = zeros(validationSetSize, yColsCount);

	for i = 1:validationSetSize

		% Input >> Hidden
		Xi = [-1 xValidationSet(i,:)]';
		Ui = inputToHiddenWeights * Xi;
		Yi = 1./(1+exp(-Ui));

		% Hidden >> Output
		Zi = [-1; Yi];
		Vi = hiddenToOutputWeights * Zi;
		Oi = 1./(1+exp(-Vi));

		% Error
		Ei = yValidationSet(i) - Oi;

		errors(i, :) = Ei;

	endfor % validation sample

	validationErrorEpoch = [
		validationErrorEpoch; 
		sum(sum(errors.^2)) / validationSetSize 
	];


	% Breakpoint checking

	currentValidationError = validationErrorEpoch(end,1);

	if currentValidationError < lastValidationError
		lastValidationError = currentValidationError;
	else
		growthControl = growthControl-1;
	endif


endwhile; % epochs


% Test