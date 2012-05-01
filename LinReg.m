function theta = LinReg(X, y, theta, alpha, num_iters)
    %%%
    % @pram X: A matrix of values like so
    % X =
    %
    %    3   7   4
    %    6   2   8
    %    4   3   2
    %
    % where each row is a training sample having the factor for x2 to xn in
    % the corresponding column. x2 to xn are the so called features.
    %
    % @param y: A vector containing target samples like so
    % y =
    %
    %    41
    %    58
    %    32
    %
    % @param theta: A vector of per feature factors used to build an
    % approximation function. The main task of this script is to modify this
    % vector iteratively. The initial theta given by the user maybe just a
    % rough estimate and can look sth like this
    % theta =
    %
    %    1
    %    1
    %    1
    %    1
    %
    % @param alpha: A value determining the step size for the iterative
    % process. Typical values (afaik) are in the range of 0.001 … 0.1.
    % 
    % @param: num_iters: The number of iterations for the approximation
    % process.

    % get matrix sizes
    [x_rows x_cols] = size(X);
    [y_rows y_cols] = size(y);
    [t_rows t_cols] = size(theta);

    % check the input
    if y_cols > 1,
        error('y should be a vector, not a matrix.');
    elseif x_rows ~= y_rows,
        error('The number of training samples does not match the number of target variables');
    elseif t_cols > 1,
        error('theta should be represented as a vector');
    elseif t_rows ~= x_cols + 1,
        error('The number of theta-values does not match the number of features (+ the one dummy row for x1).')
    end;

    % add x_{i,1}'s
    x = [ones(x_rows, 1) X];

    iter_nr = 0;
    new_theta = zeros(t_rows, 1);
    while iter_nr<num_iters,
        % iteration over the j (= x_cols + 1 = t_rows) features
        for j=1:t_rows,
            theta_j = theta(j, 1);

            i_sum = 0;
            % iteration over all the given training samples to sum up the
            % ( h_{theta}(x^(i)) -y^(i) ) * x^(i)_{j}
            for i=1:x_rows,
                sample = x(i, :)';
                h_theta_of_sample = theta' * sample;
                i_value = (h_theta_of_sample - y(i, 1)) * sample(j, 1);
                i_sum = i_sum + i_value;
            end;
            % update the theta_j and write it back to the new theta of the
            % current iteration
            theta_j = theta_j - alpha*((1/y_cols)*i_sum);
            new_theta(j, 1) = theta_j;
        end;
        theta = new_theta;
        iter_nr = iter_nr + 1;
    end;
