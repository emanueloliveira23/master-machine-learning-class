function [model] = naiveBayes(X, Y)
	
	[rowsCount, colsCount] = size(X);
	
	[countPerClass, classes] = hist(Y, unique(Y));
	
	propabilityPerClass = countPerClass./rowsCount;
	
	u = zeros(colsCount, 1);
	for c = 1:colsCount
		u(c, :) = mean(X(:, c));
	endfor 
	
	u
	
	covar = 0;
	for r = 1:rowsCount
		x = X(r, :)';
		covar = covar + ( (x-u) * (x-u)' );
	endfor
	covar = 1/(rowsCount-1) * covar;
	
	covar
	
	model.probabilities = propabilityPerClass;
	model.covar = covar;
	model.mean = u;
	
endfunction
