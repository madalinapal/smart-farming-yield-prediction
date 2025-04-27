clear all;
clc;

%% READING AND PREPROCESSING DATA  
file = "Smart_Farming_Crop_Yield_2024.csv";
[train_data, test_data] = split_data(file);

%% DEFINING VARIABLES FOR GRADIENT DESCENT METHOD
[m, n] = size(train_data);
A_1 = [train_data(:, 1:end-1), ones(m, 1)]; 
training_target = train_data(:, end); % Labels vector
learning_rate = 0.00001;
max_iter = 10000;
gradient_prag = 1e-6;
nr_neurons = 30; % Number of neurons in the hidden layer

%% GRADIENT DESCENT METHOD
[X, x_star, errors, norms, times] = gradient_descent(A_1, training_target, ...
    nr_neurons, learning_rate, max_iter, gradient_prag);

% Graphics
cumulative_times = cumsum(times); 

figure('Name', 'Gradient Descent Method', 'Color', 'w');
subplot(4, 1, 1);
semilogy(1:max_iter, errors, 'LineWidth', 1.5, 'Color', 'b');
title('Error vs. Iterations');
xlabel('Iterations');
ylabel('Error');
grid on;

subplot(4, 1, 2);
semilogy(1:max_iter, norms, 'LineWidth', 1.5, 'Color', 'r');
title('Gradient Norm vs. Iterations');
xlabel('Iterations');
ylabel('Norm');
grid on;

subplot(4, 1, 3);
plot(cumulative_times, errors, 'LineWidth', 1.5, 'Color', 'g');
title('Error vs. Time');
xlabel('Time (s)');
ylabel('Error');
grid on;

subplot(4, 1, 4);
plot(cumulative_times, norms, 'LineWidth', 1.5, 'Color', 'm');
title('Gradient Norm vs. Time');
xlabel('Time (s)');
ylabel('Norm');
grid on;

% Testing
A_testing = [test_data(:, 1:end-1), ones(size(test_data, 1), 1)]; 
testing_target = test_data(:, end);
output = cosid(A_testing * X) * x_star;

combined_vectors = horzcat(testing_target, output);
disp('Combined vectors (Testing):');
disp(combined_vectors);

mse = mean((output - testing_target).^2);
disp(['Mean Square Error for Gradient Descent Method: ', num2str(mse)]);

%% STOCHASTIC GRADIENT DESCENT METHOD

[m, n] = size(train_data);
A_1 = [train_data(:, 1:end-1), ones(m, 1)]; 
training_target = train_data(:, end); % Labels vector
learning_rate = 0.00001;
max_iter = 5000;
gradient_prag = 1e-6;
nr_neurons = 30; % Number of neurons in the hidden layer
nr_of_examples = 5;

[X, x_star, errors, norms, times] = stochastic_gradient(A_1, training_target, ...
    nr_neurons, learning_rate, max_iter, gradient_prag, nr_of_examples);

% Graphics
cumulative_times = cumsum(times); 

figure('Name', 'Stochastic Gradient Descent Method', 'Color', 'w');
subplot(4, 1, 1);
semilogy(1:max_iter, errors, 'LineWidth', 1.5, 'Color', 'b');
title('Error vs. Iterations');
xlabel('Iterations');
ylabel('Error');
grid on;

subplot(4, 1, 2);
semilogy(1:max_iter, norms, 'LineWidth', 1.5, 'Color', 'r');
title('Gradient Norm vs. Iterations');
xlabel('Iterations');
ylabel('Norm');
grid on;

subplot(4, 1, 3);
plot(cumulative_times, errors, 'LineWidth', 1.5, 'Color', 'g');
title('Error vs. Time');
xlabel('Time (s)');
ylabel('Error');
grid on;

subplot(4, 1, 4);
plot(cumulative_times, norms, 'LineWidth', 1.5, 'Color', 'm');
title('Gradient Norm vs. Time');
xlabel('Time (s)');
ylabel('Norm');
grid on;

% Testing
A_testing = [test_data(:, 1:end-1), ones(size(test_data, 1), 1)]; 
testing_target = test_data(:, end);
output = cosid(A_testing * X) * x_star;

combined_vectors = horzcat(testing_target, output);
disp('Combined vectors (Testing):');
disp(combined_vectors);

mse = mean((output - testing_target).^2);
disp(['Mean Square Error for Stochastic Gradient Descent Method: ', num2str(mse)]);
