% Weights are a matrix Nx1
function newWeights = updateWeights(oldWeights, a, e, x)
  newWeights = oldWeights(:);
  weightsCount = size(newWeights, 1);
  for i = 1:weightsCount
    % wi = wi + alpha * error * xi
    newWeights(i, 1) = oldWeights(i, 1) + a*e*x(i);
  endfor  
endfunction
