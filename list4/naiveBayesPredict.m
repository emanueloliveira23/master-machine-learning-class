# Test an Naive Bayes Model to an data sample.
function [maxScoreClass] = naiveBayesPredict(model, X)
        
    colsCount = length(X);
    classCount = size(model.classes, 1);
    
    # decision variables
    maxScore = -inf;
    maxScoreClass = -1;
    
    # loop consts
    dividerPiCols = (2*pi)^(colsCount/2);
    
    for c = 1:classCount
        
        uc = model.mean(c, :)';
        covar = reshape(model.covar(c, :), 2, 2);
        
        firstFactor = 1/(det(covar)^1/2 * dividerPiCols);
        expFactor = ( -1/2 * (X-uc)' ) * ( inv(covar) * (X-uc) );
        
        # probability of be X given C class
        pxc = firstFactor * exp( expFactor );
            
        score = pxc * model.probabilities(c);
        
        if (score > maxScore)
            maxScore = score;
            maxScoreClass = c;
        endif
        
    endfor
        
endfunction
