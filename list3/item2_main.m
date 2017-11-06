clear all;
close all;
clc;

% Load

load ex3data2.data;

ex3data2 = normalizeVector(ex3data2);

X = ex3data2(:, 1:13);
Y = ex3data2(:, 14);

[xRowsCount xColsCount] = size(X);
[yRowsCount yColsCount] = size(Y);

trainingPercent = 0.6;
validationPercent = 0.2;
testPercent = 0.2;

trainingSetSize = floor(xRowsCount * trainingPercent);
validationSetSize = floor(xRowsCount * validationPercent);
testSetSize = floor(xRowsCount * testPercent);

trainingSetStart = 1;
trainingSetEnd = trainingSetSize;
validationSetStart = trainingSetEnd + 1;
validationSetEnd = validationSetStart + validationSetSize - 1;
testSetStart = validationSetEnd + 1;
testSetEnd = testSetStart + testSetSize - 1;

allSetIndexes = randperm(xRowsCount)';
xTrainingSet = X(allSetIndexes(trainingSetStart:trainingSetEnd, :), :);
yTrainingSet = Y(allSetIndexes(trainingSetStart:trainingSetEnd, :), :);
xValidationSet = X(allSetIndexes(validationSetStart:validationSetEnd, :), :);
yValidationSet = Y(allSetIndexes(validationSetStart:validationSetEnd, :), :);
xTestSet = X(allSetIndexes(testSetStart:testSetEnd, :), :);
yTestSet = Y(allSetIndexes(testSetStart:testSetEnd, :), :);

alfa = 0.001;
hiddenLayerNeuronsCount = 1;
outputLayerNeuronsCount = yColsCount;

inputToHiddenWeights = 0.2 * rand(hiddenLayerNeuronsCount, xColsCount + 1); % + 1 >> w0
hiddenToOutputWeights = 0.2 * rand(outputLayerNeuronsCount, hiddenLayerNeuronsCount + 1); % + 1 >> w0

% Training

lastValidationError = inf;
trainingErrorEpoch = [];
validationErrorEpoch = [];

growthControl = 1;

while growthControl > 0

	% Shuffle 
	
	trainingSetIndexes = randperm(trainingSetSize)';
	xTrainingSetEpoch = xTrainingSet(trainingSetIndexes, :);
	yTrainingSetEpoch = yTrainingSet(trainingSetIndexes, :);

	% Training

	trainingErrors = zeros(trainingSetSize, yColsCount);

	for i = 1:trainingSetSize

		% Input >> Hidden
		Xi = [-1; xTrainingSetEpoch(i, :)'];
		Ui = inputToHiddenWeights * Xi;
		Yi = 1./(1+exp(-Ui));

		% Hidden >> Output
		Zi = [-1; Yi];
		Vi = hiddenToOutputWeights * Zi;

		% Error
		Ei = yTrainingSetEpoch(i) - Vi;

		% Local Gradients
		sigk = Ei;

		sigi = Yi.*(1 - Yi);
		sigi = sigi.*(hiddenToOutputWeights(:,2:end)'*sigk);

		% Update weights
		hiddenToOutputWeights = hiddenToOutputWeights + alfa*sigk*Zi';
		inputToHiddenWeights = inputToHiddenWeights + alfa*sigi*Xi';

		trainingErrors(i, :) = Ei;

	endfor % training sample

	trainingErrorEpoch = [
		trainingErrorEpoch; 
		sum(trainingErrors.^2) / trainingSetSize
	];


	% Validation

	validationErrors = zeros(validationSetSize, yColsCount);

	for i = 1:validationSetSize

		% Input >> Hidden
		Xi = [-1 xValidationSet(i,:)]';
		Ui = inputToHiddenWeights * Xi;
		Yi = 1./(1+exp(-Ui));

		% Hidden >> Output
		Zi = [-1; Yi];
		Vi = hiddenToOutputWeights * Zi;
		
		% Error
		Ei = yValidationSet(i) - Vi;

		validationErrors(i, :) = Ei;

	endfor % validation sample

	validationErrorEpoch = [
		validationErrorEpoch; 
		sum(validationErrors.^2) / validationSetSize 
	];


	% Breakpoint checking

	currentValidationError = validationErrorEpoch(end, 1);
	
	if currentValidationError > lastValidationError
		growthControl = growthControl-1;
	else
		lastValidationError = currentValidationError;
	endif

endwhile; % epochs