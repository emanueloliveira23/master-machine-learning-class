% Final Weights
disp("Pesos [w0 ... wn]: "); 
disp(res.weights);

% Plot SME x EPOCHS to TEST set
plot(1:1:EPOCHS, res.errors(:, 1));
xlabel ("Época");
ylabel ("Erro Quadrático Médio");

% Last error
disp("Last error");
disp(res.errors(EPOCHS));