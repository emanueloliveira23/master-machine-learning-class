% Function that receive training data and build a Naive Bayes Model.
% Return: Naive Bayes Model (probabilities, covariance and mean).
function [model] = naiveBayes(X, Y)
	
	[rowsCount, colsCount] = size(X);
	
	[countPerClass, classes] = hist(Y, unique(Y));
	
	classCount = length(classes);
	
	propabilityPerClass = countPerClass./rowsCount;
	
	# calculate mean of each column of data per class
	uPerClass = zeros(classCount, colsCount);
	for r = 1:rowsCount
	    x = X(r,:)';
	    y = Y(r,:);
	    for col = 1:colsCount
	    	uPerClass(y, col) += x(col,:);
	    endfor
	endfor
	for c=classes
	    uPerClass(c,:) = uPerClass(c,:) / countPerClass(c); 
	endfor
	
	% calculate covariance matrix per class
	% reshape is to use 3d matrix as 2d matrix
	covarPerClass = zeros(classCount, colsCount^2);
	for r = 1:rowsCount
	    x = X(r,:)';
	    y = Y(r,:);
	    uc = uPerClass(y, :)';
	    ux = (x-uc) * (x-uc)';
	    covarPerClass(y, :) += reshape(ux, 1, colsCount^2);
    endfor
    for c=classes	    
        clazzCovar = reshape(covarPerClass(c,:), colsCount, colsCount);
	    clazzCovar = 1/(countPerClass(c) - 1) * clazzCovar;
	    covarPerClass(c, :) = reshape(clazzCovar, 1, colsCount^2);
	endfor
		
	model.probabilities = propabilityPerClass;
	model.mean = uPerClass;
	model.covar = covarPerClass;
	model.classes = classes';
	
endfunction


