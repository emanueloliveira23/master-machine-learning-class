function res = regular_batch(x, y, lambdas)
  
  cols = size(x, 2); % features
  rows = size(x, 1);
  
  _x = [ones(rows, 1) x];
  _cols = size(_x, 2);
  
  I = eye(_cols);
  
  % Não utilizar regularização no termo w0
  I(1,1) = 0;
  
  lambdas_size = size(lambdas, 1);
  
  res = zeros(lambdas_size, _cols); % res = weights
  
  for l = 1:lambdas_size
    lambda = lambdas(l, 1);
    res(l, :) = inv(_x' * _x + lambda * I) * _x' * y;
  endfor
  
endfunction