% Plot data
admitted = [];
nonadmitted = [];

for i = 1:1:COUNT
  y = Y(i, 1);
  if y == 0
    nonadmitted = [nonadmitted; X(i, :)];
  elseif y == 1
    admitted = [admitted; X(i, :)];
  endif
endfor

figure();
plot(admitted(:, 1), admitted(:, 2), '+b');
hold on;
plot(nonadmitted(:, 1), nonadmitted(:, 2), '+r');
xlabel ("Score 1");
ylabel ("Score 2");
h = legend ({"Admittted"}, "Not Admitted");
legend (h, "location", "northeastoutside");
