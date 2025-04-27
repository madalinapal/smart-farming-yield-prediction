function [train_data, test_data] = split_data(file)
    % Read data from the specified file
    data = readtable(file, 'VariableNamingRule','preserve');
    
    % Remove irrelevant columns that are not useful for prediction
    cols_to_remove = {'farm_id', 'sensor_id', 'timestamp', ...
                      'sowing_date', 'harvest_date'};
    data = removevars(data, cols_to_remove);

    % Encode categorical variables into numeric values (label encoding)
    categorical_vars = {'region', 'crop_type', 'irrigation_type', ...
                        'fertilizer_type', 'crop_disease_status'};

    for i = 1:length(categorical_vars)
        var_name = categorical_vars{i};
        var_data = data.(var_name);  
    
        % Convert categorical or string types to cell arrays of character vectors
        if iscategorical(var_data)
            var_data = cellstr(var_data);
        elseif isstring(var_data)
            var_data = cellstr(var_data);
        end

        % Apply label encoding
        [~, ~, idx] = unique(var_data);
        data.(var_name) = idx;
    end

    % Remove any rows with missing values
    data = rmmissing(data);

    % Separate target variable (yield) from features
    target_vector = data.yield_kg_per_hectare;
    data = removevars(data, 'yield_kg_per_hectare');

    % Convert the table of features to a numerical matrix
    features_matrix = table2array(data);

    % Randomly shuffle the data for splitting
    rng(42);  % Set random seed for reproducibility
    n = size(features_matrix, 1);
    idx = randperm(n);
    train_n = round(0.8 * n);  % 80% of the data for training
    train_idx = idx(1:train_n);
    test_idx = idx(train_n+1:end);

    % Create train and test sets
    train_features = features_matrix(train_idx, :);
    test_features  = features_matrix(test_idx, :);
    train_target = target_vector(train_idx);
    test_target  = target_vector(test_idx);

    
    % Apply z-score normalization (zero mean, unit variance)
    [train_features, mu, sigma] = zscore(train_features); 
    test_features = (test_features - mu) ./ sigma; 

    [train_target, mu_target, sigma_target] = zscore(train_target);
    test_target = (test_target - mu_target) / sigma_target;


    % Combine features and target into final train and test datasets
    train_data = [train_features, train_target];
    test_data  = [test_features, test_target];
end
