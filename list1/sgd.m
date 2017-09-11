% Univariant Stochastic Gradient Descendent (SGD)
% Return an object with weighst vector and 
% Return Squred Mean Error errors by epoch
function res = sgd(x, y, alpha, epochs)
  
  x_ = x;                     % Copy of data 
  errors = zeros(epochs, 1);  % Errors by Epoch
  xlen = size(x,1);           % Rows count of X
  
  w0 = 0; % Naive initialization of weight at index 0
  w1 = 0; % Naive initialization of weight at index 1
    
  for epoch = 1:epochs
    
    x_ = x_(randperm(xlen),:); % Permuting rows of data
    
    sum_error = 0;
    
    for i = 1:xlen
      
      xi = x_(i);                     % Get data
      yi = y(i);                      % Get label
      
      yi_ = w1*xi + w0;               % Compute new label
      
      ei = yi - yi_;                  % Compute error
      
      w0 = w0 + alpha*ei;             % Update weight 0
      w1 = w1 + alpha*ei*xi;          % Update weight 1
      
      sum_error = sum_error + ei*ei;  % Accumlate squred error
      
    endfor
    
    errors(epoch,:) = sum_error / xlen; % Add Squred Mean Error
  
  endfor
  
  res.weights = [w0 w1];
  res.errors = errors;
  
endfunction

