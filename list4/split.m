function [XTraining, YTraining, XTest, YTest] = split(X, Y, trainingRate)


	[XRowsCount, XColsCount] = size(X);
	[YRowsCount, YColsCount] = size(Y);
	classCount = size(unique(Y), 1);

	# This code creates a matrix of indexes, one row by class.
	perClassRate = ceil(YRowsCount / classCount);
	idxsPerClass = zeros(classCount, perClassRate);
	for clazz = 1:classCount
		idxsClass = [];
		for idx = 1:XRowsCount
			itemClass = Y(idx, 1);
			if (itemClass == clazz)
				idxsClass = [idxsClass idx];
			endif
		endfor
		idxsPerClass(clazz, :) = idxsClass(randperm(perClassRate));
	endfor
	
	# The firsts are training
	trainingCount = ceil(XRowsCount * trainingRate);
	trainingPerClassCount = ceil(trainingCount / classCount);
	
	# The remaining is test
	
	XTraining = [];
	YTraining = [];
	XTest = [];
	YTest = [];

	for clazz = 1:classCount
		traningIdxs = idxsPerClass(clazz, 1:trainingPerClassCount);
		testIdxs = idxsPerClass(clazz, trainingPerClassCount+1:end);
		XTraining = [XTraining; X(traningIdxs, :)];
		YTraining = [YTraining; Y(traningIdxs, :)];
		XTest = [XTest; X(testIdxs, :)];
		YTest = [YTest; Y(testIdxs, :)];
	endfor

endfunction
