function [X, x_star, errors, norms, times] = gradient_descent(A, e, m, ...
    learning_rate, max_iter, prag_gradient)

    [lines, columns] = size(A);
    X = 0.01 * randn(columns, m);  % Hidden layer weights
    x_star = 0.01 * randn(m, 1);    % Output layer weights

    errors = zeros(1, max_iter);    
    norms = zeros(1, max_iter);      
    times = zeros(1, max_iter);      

    gradient_norm = inf; 
    iter = 0;              

    while iter < max_iter && gradient_norm > prag_gradient
        iter = iter + 1;
        tic;  

        % Forward pass
        hidden_output = cosid(A * X);          
        predicted_output = hidden_output * x_star;  
        error = predicted_output - e;          

        % Backpropagation
        hidden_output_derivat = cosid_deriv(A * X);  

        % Gradient computations 
        dL_dx_star = (hidden_output' * error) / lines;  
        dL_dX = (A' * (hidden_output_derivat .* (error * x_star'))) / lines;  

        % L2 Regularization (prevents overfitting)
        lambda = 0.001;
        dL_dX = dL_dX + lambda * X;           % Add regularization term for X
        dL_dx_star = dL_dx_star + lambda * x_star;  % Add regularization term for x_star

        % Parameter updates
        X = X - learning_rate * dL_dX;         % Update hidden layer weights
        x_star = x_star - learning_rate * dL_dx_star;  % Update output layer weights

        gradient_norm = norm([dL_dX(:); dL_dx_star]);

        % Track performance metrics
        times(iter) = toc;                     % Record iteration time
        errors(iter) = sum(error.^2) / (2 * lines);  % MSE loss (with 1/2 term)
        norms(iter) = norm([dL_dX(:); dL_dx_star]);  % Combined gradient norm

        % Progress reporting
        if mod(iter, 1000) == 0
            fprintf('Iteration %d: Error = %.6f, Norm = %.6f\n', iter, errors(iter), norms(iter));
        end
    end

    % Trim unused preallocated space
    errors = errors(1:iter);
    norms = norms(1:iter);
    times = times(1:iter);
end