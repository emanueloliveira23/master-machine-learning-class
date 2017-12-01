close all;
clear all;
clc;

% Load data
load ex5data1.data;
X = normalizeVector(ex5data1(:,1:4));
Y = ex5data1(:, 5);
[sizeX, colsX] = size(X);

% Drop means
for c = 1:colsX
  m = mean(X(:, c));
  X(:, c) -= m;
endfor

% Covariance matrix
covX = cov(X);

% Eigenvector and Eigenvalues
[eigvecCovX, eigvalCovX] = eig(covX);


% Get top K eigenvalues
K = 2;
eigvalCovX = diag(eigvalCovX);
topkEigValCovX = sort(eigvalCovX, 'descend')(1:K, :);

% Get top K eigenvector 
topkEigVecCovX = []; % Eigenvectors of top k eigenvalues
for k = 1:K
  eigVal = topkEigValCovX(k,:);
  eigValIdx = find(eigvalCovX == eigVal);
  topkEigVecCovX = [topkEigVecCovX; eigvecCovX(eigValIdx, :)];
endfor
topkEigVecCovX;    

% Reducing matrix
rX = X * topkEigVecCovX';

plot(rX(1:50,1), rX(1:50,2), '+r');
hold on;
plot(rX(51:100,1), rX(51:100,2), '+g');
hold on;
plot(rX(101:150,1), rX(101:150,2), '+b');
hold on;
legend('setosa', 'versicolo', 'virginica');