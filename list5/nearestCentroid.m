function c = nearestCentroid(x, means)

  k = size(means, 1);
  ds = zeros(k, 1);
  for mk = 1:k
    ds(mk, 1) = distance(x, means(mk, :));
  endfor
  c = find( ds == min(ds) );

endfunction 