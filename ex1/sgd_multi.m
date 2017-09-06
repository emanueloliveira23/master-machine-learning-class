% Multivariant Stochastic Gradient Descendent (SGD)
% Return an object with weighst vector and 
% Return Squred Mean Error errors by epoch
function res = sgd_multi(x, y, alpha, epochs)
  
  % Copy of data 
  x_ = x;   

  % Errors by Epoch
  errors = zeros(epochs, 1);

  % Rows count of X
  x_size = size(x,1);         

  % Columns (attributes) count of X
  attrs_cnt = size(x, 2);
  
  % Initialization of weights
  % '+ 1' becasue weight at index 0
  weights = zeros(1 + attrs_cnt, 1);
  
  for epoch = 1:epochs
    
    % Permuting rows of data
    x_ = x_(randperm(x_size),:);
    
    sum_error = 0;
    
    for i = 1:x_size
      
      % Get data (x) and label (y)
      xi = [1 x_(i)];
      yi = y(i);
      
      % Compute new label
      result_yi = xi * weights; % w0*1 + w1*x1 + ... + wn*xn; 
      
      % Compute error
      error = yi - result_yi;
      
      % Update weights
      weights = update_weights(weights, alpha, error, xi); 
      
      % Accumlate squred error
      sum_error = sum_error + error*error; 
      
    endfor
    
    % Add Squred Mean Error
    errors(epoch,1) = sum_error / x_size; 
  
  endfor
  
  % Wrap response
  res.weights = weights;
  res.errors = errors;
  
endfunction

