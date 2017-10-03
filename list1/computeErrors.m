function errors = computeErrors(x, y, w)
  
  count = size(x, 1);
  fullX = [ones(count, 1) x];
  weightsCount = size(W, 1);
  errors = zeros(weightsCount, 1);
  
  for i = 1:weightsCount
    error = 0;
    wi = w(i, :)';
    
    for j = 1:count
      xj = fullX(j, :);
      yj = y(j, :);
      e = yj - xj * wi;
      error = error + e*e;
    endfor
    
    errors(l, 1) = error / count;
    
  endfor 

endfunction