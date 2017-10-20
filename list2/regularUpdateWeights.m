% Weights are a matrix Nx1
function newWeights = regularUpdateWeights(oldWeights, alfa, error, x, lambda)
  %weightsCount = size(oldWeights, 1);
  %newWeights = oldWeights(weightsCount, 1);
  
  % update first
  %newWeights(1, 1) = oldWeights(1, 1) + a*error*x(1);
  
  % update rest
  %for j = 2:weightsCount
  %  % wj = wj + a*e*xj
  %  oldWeight = oldWeights(j, 1);
  %  newWeights(j, 1) = oldWeight + a*(error*x(j) - lambda*oldWeight);
  %endfor
  
  weightsCount = size(oldWeights, 1);
  
  I = eye(weightsCount);
  I(1,1) = 0; 
  I = I*lambda;
  
  newWeights = oldWeights + alfa*(error*x(:) - I*oldWeights);
  
endfunction
