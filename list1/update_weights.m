% Update a weights vector
function new_weights = update_weights(old_weights, alpha, error, x)
  
  new_weights = old_weights(:);
  weights_cnt = size(new_weights, 1);

  for i = 1:weights_cnt
    % wi = wi + alpha * error * xi
    new_weights(i, 1) = old_weights(i, 1) + alpha * error * x(i);
  endfor

endfunction
