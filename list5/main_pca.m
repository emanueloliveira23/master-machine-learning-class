close all;
clear all;
clc;

% Load data
load ex5data1.data;
X = ex5data1(:,1:4);
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

eigvecCovX
eigvalCovX

K = 2;

% Get top K eigenvalues
eigvalCovX = diag(eigvalCovX);
topkEigValCovX = sort(eigvalCovX, 'descend')(1:K, :);

% Get top K eigenvector 
topkEigVecCovX = [];
topkEigVecCovXIdx = [];
for k = 1:K
  eigVal = topkEigValCovX(k,:);
  eigValIdx = find(eigvalCovX == eigVal);
  topkEigVecCovX = [topkEigVecCovX; eigvecCovX(eigValIdx, :)];
  topkEigVecCovXIdx = [topkEigVecCovXIdx; eigValIdx];
endfor
topkEigVecCovX % Eigenvectors of top k eigenvalues
topkEigVecCovXIdx % Indexes of columns of top k eigenvectors