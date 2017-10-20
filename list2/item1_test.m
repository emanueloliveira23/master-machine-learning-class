% Final Weights
disp("Weights [w0 ... wn]: ");
disp(res.weights);

% Plot SME x EPOCHS to TEST set
figure();
plot(1:1:EPOCHS, res.errors(:, 1));
xlabel ("Epoch");
ylabel ("Squared Mean Error");

% Last error
disp("Last error");
disp(res.errors(EPOCHS));
