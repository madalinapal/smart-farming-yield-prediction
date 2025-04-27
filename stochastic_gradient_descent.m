function [X, x_star, losses, gradients, times] = metoda_gradient_stochastic(...
    A, e, n, m, rata_de_invatare, max_iter, prag_gradient, nr_exemple)

    % Initializare parametri
    N = size(A, 1);
    X = randn(n + 1, m) * 0.01;
    x_star = randn(m, 1) * 0.01;

    iter = 0;
    norma_gradient = inf;

    % Vectori pentru loss, norma gradientului si timp
    losses = zeros(1, max_iter);
    gradients = zeros(1, max_iter);
    times = zeros(1, max_iter);

    epsilon = 1e-8;

    while iter < max_iter && norma_gradient > prag_gradient
        iter = iter + 1;
        t_start = tic;

        % Selectare exemple random
        idx = randperm(N, nr_exemple);
        A_batch = A(idx, :);
        e_batch = e(idx);

        % Forward pass
        Z = A_batch * X;
        G = cos(Z) - Z;
        y = G * x_star;
        y_prob = 1 ./ (1 + exp(-y)); % sigmoid

        % Loss (cross-entropy)
        loss = -mean(e_batch .* log(y_prob + epsilon) + ...
                     (1 - e_batch) .* log(1 - y_prob + epsilon));
        losses(iter) = loss;

        % Backpropagation
        delta_L = (y_prob - e_batch) / nr_exemple;
        dL_dx = G' * delta_L;

        dG_dZ = -sin(Z) - 1;
        delta_Z = delta_L .* x_star' .* dG_dZ;

        dL_dX = A_batch' * delta_Z;

        % Norma gradient
        norma_gradient = norm([dL_dX(:); dL_dx]);
        gradients(iter) = norma_gradient;

        % Update parametri
        X = X - rata_de_invatare * dL_dX;
        x_star = x_star - rata_de_invatare * dL_dx;

        times(iter) = toc(t_start);

        if mod(iter, 1000) == 0
            fprintf('Iter %d: Loss=%.4f, ||∇||=%.4e\n', iter, loss, norma_gradient);
        end
    end

    % Taiere vectori
    losses = losses(1:iter);
    gradients = gradients(1:iter);
    times = times(1:iter);
end
