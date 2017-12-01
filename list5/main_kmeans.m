close all;
clear all;
clc;

% Load data
load ex5data1.data;
X = normalizeVector(ex5data1(:,1:4));
Y = ex5data1(:, 5);
sizeX = size(X, 1);

% First approach: Ad-Hoc
error = zeros(4, 1);

for k = 2:5

  means = kmeans(X, k);

  % Compute errors
  sse = 0;
  for xi = 1:sizeX  
    x = X(xi, :);
    centroid = nearestCentroid(x, means);
    centroid = means(centroid, :);
    d = distance(centroid, x);
    sse += d^2;
  endfor

  error(k-1, :) = sse;

endfor

plot(2:5, error)


k = 3;
means = kmeans(X, k);

classes = [1:50; 51:100; 101:150];
for c = 1:3
  
  xs = X(classes(c,:), :);
  ys = [];
  for xi = 1:50
    x = xs(xi, :);
    ys = [ys; nearestCentroid(x, means)];
  endfor

  [freq, groups]=hist(ys,unique(ys))

endfor