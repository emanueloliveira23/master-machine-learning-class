function [matrix] = confusionMatrix(model, classifier, X, y)

    classCount = size(model.classes, 1);
    
    matrix = zeros(classCount, classCount);
    
    rowsCount = size(X, 1);
    
    for r = 1:rowsCount
        
        x = X(r,:)';
        yReal = y(r,:);
        yPredict = classifier(model, x);
        
        matrix(yReal, yPredict) += 1;
        
    endfor

endfunction
