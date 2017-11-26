% Test an Gaussian Quadradict Discriminant to an data sample.
function [maxScoreClass] = quadraticDiscriminantPredict(model, X)
        
    colsCount = length(X);
    classCount = size(model.classes, 1);

    for c = 1:classCount
        covar = reshape(model.covar(c, :), colsCount, colsCount);
        covar = diag(diag(covar));
        model.covar(c, :) = reshape(covar, 1, colsCount^2);
    endfor

    maxScoreClass = naiveBayesPredict(model, X);

endfunction
