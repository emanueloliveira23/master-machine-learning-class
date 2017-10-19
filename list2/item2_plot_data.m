function item2_plot_data(X, Y)

  % Plot data
  approved = [];
  reproved = [];

  for i = 1:1:size(X, 1)
    y = Y(i, 1);
    if y == 0
      reproved = [reproved; X(i, :)];
    elseif y == 1
      approved = [approved; X(i, :)];
    endif
  endfor

  plot(approved(:, 1), approved(:, 2), '+b');
  hold on;
  plot(reproved(:, 1), reproved(:, 2), '+r');
  xlabel ("Score 1");
  ylabel ("Score 2");
  h = legend ({"+ Aprovados"}, "+ Reprovados");
  legend (h, "location", "northeastoutside");

endfunction