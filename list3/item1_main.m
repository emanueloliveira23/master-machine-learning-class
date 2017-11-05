close all;
clear all;
clc;

% Load


load ex3data1.mat;

% load X matrix
% load T matrix

[xRowsCount xColsCount] = size(X);
[yRowsCount yColsCount] = size(T);

traningPercent = 0.8;
validationPercent = 0.1;
testPercent = 0.1;

trainingSetSize = floor(xRowsCount * traningPercent);
validationSetSize = floor(xRowsCount * validationPercent);
testSetSize = floor(xRowsCount * testPercent);

trainingSetStart = 1;
trainingSetEnd = trainingSetSize;
validationSetStart = trainingSetEnd + 1;
validationSetEnd = validationSetStart + validationSetSize - 1;
testSetStart = validationSetEnd + 1;
testSetEnd = testSetStart + testSetSize - 1;

allSetIndexes = randperm(xRowsCount)';
xTraningSet = X(allSetIndexes(trainingSetStart:trainingSetEnd, :), :);
yTraningSet = T(allSetIndexes(trainingSetStart:trainingSetEnd, :), :);
xValidationSet = X(allSetIndexes(validationSetStart:validationSetEnd, :), :);
yValidationSet = T(allSetIndexes(validationSetStart:validationSetEnd, :), :);
xTestSet = X(allSetIndexes(testSetStart:testSetEnd, :), :);
yTestSet = T(allSetIndexes(testSetStart:testSetEnd, :), :);


alpha = 0.0001;
hiddenLayerNeuronsCount = 20;
outputLayerNeuronsCount = 10;

inputToHiddenWeights = 0.2 * rand(hiddenLayerNeuronsCount, xColsCount + 1); % + 1 >> w0
hiddenToOutputWeights = 0.2 * rand(outputLayerNeuronsCount, hiddenLayerNeuronsCount + 1); % + 1 >> w0

% Traning

lastValidationError = inf;
trainingErrorEpoch = [];
validationErrorEpoch = [];

growthControl = 1;

while growthControl > 0

	% Shuffle 
	
	traningSetIndexes = randperm(trainingSetSize)';
	xTraningSetEpoch = xTraningSet(traningSetIndexes, :);
	yTraningSetEpoch = yTraningSet(traningSetIndexes, :);

	validationSetIndexes = randperm(validationSetSize)';
	xValidationSetEpoch = xValidationSet(validationSetIndexes, :);
	yValidationSetEpoch = yValidationSet(validationSetIndexes, :);


	% Traning

	trainingErrors = zeros(trainingSetSize, yColsCount);

	for i = 1:trainingSetSize

		% Input >> Hidden
		Xi = [-1 xTraningSetEpoch(i,:)]';
		Ui = inputToHiddenWeights * Xi;
		Yi = 1./(1+exp(-Ui));

		% Hidden >> Output
		Zi = [-1; Yi];
		Vi = hiddenToOutputWeights * Zi;
		Oi = 1./(1+exp(-Vi));

		% Error
		Ei = yTraningSetEpoch(i) - Oi;

		% Local Gradients
		sigk = Oi.*(1 - Oi);
		sigk = Ei.*sigk;

		sigi = Yi.*(1 - Yi);
		sigi = sigi.*(hiddenToOutputWeights(:,2:end)'*sigk);

		% Update weights
		hiddenToOutputWeights = hiddenToOutputWeights + alpha*sigk*Zi';
		inputToHiddenWeights = inputToHiddenWeights + alpha*sigi*Xi';

		trainingErrors(i, :) = Ei;

	endfor % traning sample

	trainingErrorEpoch = [
		trainingErrorEpoch; 
		sum(sum(trainingErrors.^2)) / trainingSetSize
	];


	% Validation

	validationErrors = zeros(validationSetSize, yColsCount);

	for i = 1:validationSetSize

		% Input >> Hidden
		Xi = [-1 xValidationSetEpoch(i,:)]';
		Ui = inputToHiddenWeights * Xi;
		Yi = 1./(1+exp(-Ui));

		% Hidden >> Output
		Zi = [-1; Yi];
		Vi = hiddenToOutputWeights * Zi;
		Oi = 1./(1+exp(-Vi));

		% Error
		Ei = yValidationSetEpoch(i) - Oi;

		validationErrors(i, :) = Ei;

	endfor % validation sample

	validationErrorEpoch = [
		validationErrorEpoch; 
		sum(sum(validationErrors.^2)) / validationSetSize 
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