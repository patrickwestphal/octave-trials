function [centroids, clusterAssignment] = kMeans(X, nrCluster, num_iters)
    %%%
    % @param X: A n x m matrix of n m-dimensional values (--> n value lines
    % each having m different features)
    %
    % @param nrCluster: the number of clusters to put the n values into
    % 
    % @param num_iters: the number of iterations the whole algorithm will go
    % through
    
    [num_samples num_features] = size(X);

    %%% 1)  init centroids vector
    sample_maxima = max(X); % get the per colum maxima
    centroids = zeros(nrCluster, num_features);
    % select initial centroids randomly
    for j=1:nrCluster,
        for i=1:num_features,
            centroids(j, i) = rand()*sample_maxima(1,i);
        end;
    end;
    %%% done.


    % init cluster assignment matrix
    clusterAssignment = zeros(num_samples, 1);

    iter_nr = 0;
    while iter_nr<num_iters,
        % iterate over samples
        for k=1:num_samples,
            sample = X(k, :);
            distances = zeros(nrCluster, 1);

            % iterate over centroids to get the different distances between
            % the centroids and the sample values
            for j=1:nrCluster,
                sum_of_squares = 0;

                % calculate the Euclidean distance
                for i=1:num_features,
                    tmp = (sample(i)-centroids(j,i))^2;
                    sum_of_squares = sum_of_squares + tmp;
                end;
                distances(j, 1) = sqrt(sum_of_squares);
                % done
            end;

            %%% 2)  get the cluster with the minimal (Euclidean) distance
            min_dist = min(distances);
            index = find(distances == min_dist);
            clusterAssignment(k, 1) = index;
            %%% done.
        end;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % just for debugging an X with only two features five clusters
         close;
         % plot centroids
         plot(centroids(1, 1), centroids(1, 2), 'ob');
         hold on;
         plot(centroids(2, 1), centroids(2, 2), 'or');
         plot(centroids(3, 1), centroids(3, 2), 'og');
         plot(centroids(4, 1), centroids(4, 2), 'om');
         plot(centroids(5, 1), centroids(5, 2), 'oc');

         % plot values
         for i=1:60,
             if clusterAssignment(i,1) == 1,
                 plot(X(i, 1), X(i, 2), 'xb');
             elseif clusterAssignment(i,1) == 2,
                 plot(X(i, 1), X(i, 2), 'xr');
             elseif clusterAssignment(i,1) == 3,
                 plot(X(i, 1), X(i, 2), 'xg');
             elseif clusterAssignment(i,1) == 4,
                 plot(X(i, 1), X(i, 2), 'xm');
             elseif clusterAssignment(i,1) == 5,
                 plot(X(i, 1), X(i, 2), 'xc');
             end;
         end;
         filename = sprintf('kmean-%02i', iter_nr);
         title_str = sprintf('5-Means clustering after the %i. iteration', iter_nr);
         title(title_str);
         legend('centeroid 1', 'centeroid 2', 'centeroid 3', 'centeroid 4', 'centeroid 5');
         print(filename, '-dpng');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        %%% 3)  calculate new centeroid positions
        % init nrCluster x num_features matrix intended to hold the temporary
        % sums of the per category feature values
        centroid_sums = zeros(nrCluster, num_features);
        centroid_counts= zeros(nrCluster, num_features);

        % sum up all the feature values per associated cluster
        for j=1:num_samples,
            cluster_nr = clusterAssignment(j);
            for i=1:num_features,
                centroid_sums(cluster_nr, i) = centroid_sums(cluster_nr, i) + X(j,i);
                centroid_counts(cluster_nr, i) = centroid_counts(cluster_nr, i) + 1;
            end;
        end;

        % now get the mean of these sums (per cluster, of course)
        for j=1:nrCluster,
            for i=1:num_features,
                centroids(j, i) = centroid_sums(j, i) / centroid_counts(j, i);
            end;
        end;
        %%% done.

        iter_nr = iter_nr + 1;
    end;
