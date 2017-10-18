function vn = normalizeVector(v)
  
  vRows = size(v, 1);
  vCols = size(v, 2);
  vn = zeros(vRows, vCols);
  
  for col=1:vCols
    
    vColVals = v(:, col);
    
    vMax = max(vColVals);
    vMin = min(vColVals);
    vRange = vMax - vMin;
    
    for row=1:vRows
      vi = vColVals(row, 1);
      vn(row, col) = (vi - vMin) / vRange;
    endfor
  
  endfor

endfunction