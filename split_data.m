function [testing_data, training_data, nr_training_examples] = split_data(file)

%Loading the database
ds = readtable(file);

%Training data percentage
training_percentage = 0.8;

%Total number of data
N = size(ds, 1);

nr_training_examples = floor(training_percentage * N);

%amestec baza de date pt a nu avea bias??
rng(42);
ds = ds(randperm(N, :));

end