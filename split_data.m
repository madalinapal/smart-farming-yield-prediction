function [train_data, test_data] = split_data(file, normalize)
    data = readtable(file, 'VariableNamingRule','preserve');
    
    % Eliminating irrelevant columns
    cols_to_remove = {'farm_id', 'sensor_id', 'timestamp', ...
                      'sowing_date', 'harvest_date'};
    data = removevars(data, cols_to_remove);

    % Processing categorical variables (using label encoding)
    categorical_vars = {'region', 'crop_type', 'irrigation_type', ...
                        'fertilizer_type', 'crop_disease_status'};

    for i = 1:length(categorical_vars)
        var_name = categorical_vars{i};
        var_data = data.(var_name);  
    
        if iscategorical(var_data)
            var_data = cellstr(var_data);
        elseif isstring(var_data)
            var_data = cellstr(var_data);
        end
        [~, ~, idx] = unique(var_data);
        data.(var_name) = idx;
    end

    % Eliminating missing values lines
    data = rmmissing(data);

    target_vector = data.yield_kg_per_hectare;
    data = removevars(data, 'yield_kg_per_hectare');

    features_matrix = table2array(data);

    rng(42); 
    n = size(features_matrix, 1);
    idx = randperm(n);
    train_n = round(0.8 * n);
    train_idx = idx(1:train_n);
    test_idx = idx(train_n+1:end);

    train_features = features_matrix(train_idx, :);
    test_features  = features_matrix(test_idx, :);
    train_target = target_vector(train_idx);
    test_target  = target_vector(test_idx);

    % Normalizing data
    if nargin < 2
        normalize = true; 
    end

    if normalize
        [train_features, mu, sigma] = zscore(train_features); 
        test_features = (test_features - mu) ./ sigma; 
        [train_target, mu_target, sigma_target] = zscore(train_target);
        test_target = (test_target - mu_target) / sigma_target;

    end

    train_data = [train_features, train_target];
    test_data  = [test_features, test_target];
end
