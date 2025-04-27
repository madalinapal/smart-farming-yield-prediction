function [X, x_star, errors, norms, times] = stochastic_gradient(A, e, m, ...
    learning_rate, max_iter, prag_gradient, nr_of_examples)

    % Initialization of parameters
    [lines, columns] = size(A);
    X = 0.01 * randn(columns, m); % Initialize weights for the hidden layer
    x_star = 0.01 * randn(m, 1);  % Initialize weights for the output layer
    iter = 0;
    gradient_norm = inf;

    % Defining vectors to store error, gradient norm, and time at each iteration
    errors = zeros(1, max_iter);
    norms = zeros(1, max_iter);
    times = zeros(1, max_iter);

    % Stochastic Gradient Descent Loop
    while iter < max_iter && gradient_norm > prag_gradient

        iter = iter + 1;
        tic; % Start timing for current iteration

        % Randomly select a batch of training examples
        idx = randperm(lines, nr_of_examples);

        % Forward propagation
        hidden_output = cosid(A(idx, :) * X); % Hidden layer output after activation
        predicted_output = hidden_output * x_star; % Final output prediction

        % Compute error
        error = predicted_output - e(idx);    

        % Compute gradients
        dL_dX = zeros(size(X));
        dL_dx_star = zeros(size(x_star));

        hidden_output_derivat = cosid_deriv(A(idx, :) * X); % Derivative of hidden activation

        % Compute gradients using only the selected batch
        dL_dx_star = (hidden_output' * error) / nr_of_examples;  
        dL_dX = (A(idx, :)' * (hidden_output_derivat .* (error * x_star'))) / nr_of_examples; 

        % Calculate the norm of the gradient (for convergence check)
        gradient_norm = norm([dL_dX(:); dL_dx_star]);

        % Update parameters
        X = X - learning_rate * dL_dX;
        x_star = x_star - learning_rate * dL_dx_star;

        % Store elapsed time
        times(iter) = toc;

        % Store error and gradient norm
        errors(iter) = sum(error.^2) / 2;
        norms(iter) = gradient_norm;

        % Display progress every 500 iterations
        if mod(iter, 500) == 0
            fprintf('Iteration: %d: Error = %f, Gradient Norm = %f\n', ...
                iter, errors(iter), gradient_norm);
        end
    end

end
