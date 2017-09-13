function res = regular_batch(x, y, lambdas)
  
  rows = size(x, 1); % sample
  cols = size(x, 2); % features
  lambdas_size = size(lambdas, 1);
  
  _x = [ones(rows, 1) x];
  cols_size = size(_x, 2);
  I = eye(cols_size);
  
  res = zeros(lambdas_size, cols_size); % res = weights
  for l = 1:lambdas_size
    lambda = lambdas(l, 1);
    lambdaI = lambda * I;
    lambdaI(1,1) = 0;  % "Não utilizar regularização no termo w0"
    res(l, :) = inv(_x' * _x + lambdaI) * _x' * y;
  endfor
  
endfunction