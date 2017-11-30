function means = kmeans(X, k)

  [sizeX, colsX] = size(X);

  % K-Means algorithm

  % Choose k intial centroids
  means = X(randperm(sizeX, k), :);

  % Loop: while has change
  change = true;
  while change

    newMeansSum = zeros(k, colsX);
    newMeansCount = zeros(k, 1);

    for xi = 1:sizeX

      x = X(xi, :);
      
      % Find nearest centroid
      centroid = nearestCentroid(x, means);

      % Assign min group
      newMeansSum(centroid, :) += x;
      newMeansCount(centroid, :) += 1;
      
    endfor

    % Compute new means
    newMeans = newMeansSum./newMeansCount;
    
    % Check the change
    change = sum(newMeans == means) != k;
    means = newMeans(:,:);

  endwhile

endfunction