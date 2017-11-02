

epochs = size(trainingErrorEpoch, 1);

figure();

plot(1:epochs, trainingErrorEpoch, 'r');

hold on;

plot(1:epochs, validationErrorEpoch, 'b');

xlabel ("Epoch");

ylabel ("Squared Mean Error");

h = legend ({"Traning"}, "Validation");

legend (h, "location", "northeastoutside");