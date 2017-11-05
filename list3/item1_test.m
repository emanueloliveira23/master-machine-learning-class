
% Test

testErrors = zeros(testSetSize, yColsCount);

for i = 1:testSetSize

	% Input >> Hidden
	Xi = [-1 xTestSet(i,:)]';
	Ui = inputToHiddenWeights * Xi;
	Yi = 1./(1+exp(-Ui));

	% Hidden >> Output
	Zi = [-1; Yi];
	Vi = hiddenToOutputWeights * Zi;
	Oi = 1./(1+exp(-Vi));

	% Error
	Ei = yTestSet(i) - Oi;

	testErrors(i, :) = Ei;

endfor % validation sample

testError = sum(sum(testErrors.^2)) / testSetSize;

disp('Test error: '), disp(testError);