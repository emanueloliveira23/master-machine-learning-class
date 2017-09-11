function res = batch(x, y)
  
  _x = [ones(size(x, 1), 1) x];
  res = inv(_x' * _x) * _x' * y;

endfunction