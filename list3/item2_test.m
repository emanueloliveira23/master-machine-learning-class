testPredictions = zeros(testSetSize, 1);

for i = 1:testSetSize

	% Input >> Hidde
	Xi = [-1 xTestSet(i,:)]';
	Ui = inputToHiddenWeights * Xi;
	Yi = 1./(1+exp(-Ui));

	% Hidden >> Output
	Zi = [-1; Yi];
	prediction = hiddenToOutputWeights * Zi;

	testPredictions(i, :) = prediction;

endfor % validation sample

figure();

plot(1:testSetSize, yTestSet, 'r');

hold on;

plot(1:testSetSize, testPredictions, 'b');

xlabel ("House");

ylabel ("Price");

h = legend ({"Real"}, "Prediction");

legend (h, "location", "northeastoutside");