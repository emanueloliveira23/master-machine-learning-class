function d = distance(x1, x2)

  % Euclidian distance
  d = sqrt( sum( (x1 - x2).^2 ) );

endfunction