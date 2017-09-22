clear all;
close all;
clc;

load ex1data3.txt
x = ex1data3(:, 1:5);
y = ex1data3(:, 6);

training_size = 30;
test_size = size(x, 1) - training_size;

training_x = x(1:training_size, :);
training_y = y(1:training_size, :);

test_x = x(training_size+1:training_size+test_size, :);
test_y = y(training_size+1:training_size+test_size, :);

lambdas = [0; 1; 2; 3; 4; 5];
lambda_weights = regular_batch(training_x, training_y, lambdas);

% Item 1: print weights by lambda
disp(lambda_weights);

% Item 2: lambda x Square Mean Erro in training and test sets

% Util function
function errors = sme(tx, ty, weights)
  tx_size = size(tx, 1);
  weights_size = size(weights, 1);
  errors = zeros(weights_size, 1);
  
  for l = 1:weights_size
    error_sum = 0;
    w = weights(l, :)';
    
    for i = 1:tx_size
      txi = [1 tx(i, :)];
      tyi = ty(i, 1);
      te = tyi - (txi * w);
      error_sum = error_sum + (te * te);
    endfor
    
    errors(l, 1) = error_sum / tx_size;
    
  endfor 

endfunction

% Item 2.1: lambda x Square Mean Erro in training
training_errors = sme(training_x, training_y, lambda_weights);
figure(1);
plot(lambdas, training_errors);


% Item 2.2: lambda x Square Mean Erro in test
test_errors = sme(test_x, test_y, lambda_weights); 
figure(2);
plot(lambdas, test_errors, 'r');

% Item extra: plotting the 2 graphs in same figure
figure(3);
plot(lambdas, training_errors, 'b');
hold on;
plot(lambdas, test_errors, 'r');