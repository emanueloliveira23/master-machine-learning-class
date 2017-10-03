% Multivariant Stochastic Gradient Descendent (SGD)
function res = stochasticGradDesc(x_training, y_training, x_test, y_test, a, epochs)

  trainingCount = size(x_training,1); % Rows count of X training
  
  testCount = size(x_test, 1);        % Rows count of X test
  
  attrsCount = size(x_training, 2);   % Columns (attributes) count of X
  
  weights = zeros(1 + attrsCount, 1); % Init of weights
                                      % '1 +' becasue weight at index 0
  
  errors = zeros(epochs, 1);          % Init of errors by epoch
  
  for epoch = 1:epochs
    
    idxs = randperm(trainingCount);                 %
    for i = 1:trainingCount                         %
      idx = idxs(i);                                %
      xi = [1 x_training(idx, :)]';                 %
      yi = y_training(idx, :);                      % Traning
      yi_ = 1 / (1 + e**(-weights' * xi));          %
      ei = yi - yi_;                                %
      weights = updateWeights(weights, a, ei, xi);  %
    endfor                                          %
    
    error = 0;                                      %
    for i = 1:testCount                             %
      xi = [1 x_test(i, :)]';                       %
      yi = y_test(i, :);                            %
      yi_ = 1 / (1 + e**(-weights' * xi));          %
      ei = yi - yi_;                                % Test (computing error)
      error = error + ei*ei;                        %
    endfor                                          %
    error = error / testCount;                      %
    errors(epoch, :) = error;                       %
    
  endfor
  
  res.weights = weights;  % result weights
  res.errors = errors;    % result error (and error per epoch)
    
endfunction

