function res = regular_batch(x, y, lambdas)
  
  n = size(x,1);
  m = size(x,2);
  I = eye(m, n);
  
  x_size = size(x, 1);
  lambdas_size = size(lambdas, x_size);
  
  res.weights = zeros(lambdas_size, );
  for l = 1:
    lambda = lambdas(l, 1);
    res.weights(l, :) = inv(x'x + lambda * I) * x'y
  endfor
  
endfunction