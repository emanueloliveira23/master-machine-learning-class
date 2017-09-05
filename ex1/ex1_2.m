% Univariant Stochastic Gradient Descendent
function res = sgd(x, y, a=0.001, epochs=1000)
  
  res = zeros(epochs, 2)
  xlen = length(x);
  
  for epoch = 1:epochs
    
    w0 = 0; % Naive initialization
    w1 = 0; % Naive initialization
    data = permute(zeros(xlength, 1), x);
    
    for i = 0:xlen
      
      xi = x(i);          % Get data
      yi = y(i);          % Get label
      
      yi_ = w1*xi + w0;   % Compute new label
      
      ei = yi - yi_;      % Compute error
      
      w0 = w0 + a*ei;     % Update weight 0
      w1 = w1 + a*ei*xi;  % Update weight 1
      
    endfor
    
    res(epoch) = [w0 w1]
  
  endfor
  
endfunction

