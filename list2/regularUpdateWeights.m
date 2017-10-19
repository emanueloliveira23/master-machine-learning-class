% Weights are a matrix Nx1
function newWeights = regularUpdateWeights(oldWeights, a, e, x, lambda)
  newWeights = oldWeights(:);
  weightsCount = size(newWeights, 1);
  
  % update first
  newWeights(1, 1) = oldWeights(1, 1) + a*e*x(1);
  
  % update rest
  for j = 2:weightsCount
    % wj = wj + a*e*xj
    oldWeight = oldWeights(j, 1);
    newWeights(j, 1) = oldWeight + a*(e*x(j) - lambda*oldWeight);
  endfor
endfunction
