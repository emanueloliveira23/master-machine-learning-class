% Multivariant Stochastic Gradient Descendent (SGD)
function weights = regularStochasticGradDesc(X, y, alpha, epochs, lambda)

  xSize = size(X,1); % Rows count of X training
   
  weights = zeros(size(X, 2), 1); % Init of weights
    
  for epoch = 1:epochs
    
    idxs = randperm(xSize);
    
    for i = 1:xSize
      idx = idxs(i);
      xi = X(idx, :)';
      yi = y(idx, :);
      resYi = 1 / (1 + e**(-weights' * xi));
      ei = yi - resYi;
      weights = regularUpdateWeights(weights, alpha, ei, xi, lambda);
    endfor
    
  endfor
    
endfunction