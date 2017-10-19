function [stat, B_best, f_total_best ] = init_B_by_kmeans( X, k, T, mu, emb_method )
%INIT_B_BY_KMEANS Summary of this function goes here
% Detailed explanation goes here
% X: set of samples
% k: number of centroids
% T: number of iteration

B_best = 0;
f_total_best = realmax;
n = size(X, 2);

f_total_locality_best = realmax;
f_total_reconstruction_best = realmax;
gama_all_best = [];

% Set number of sampling used for initilization with Kmeans
% n_ids = 10^5; 
n_ids = 10^6;  
if (n_ids >= n)
    n_ids = n;
    ids = 1:n;
else
    ids = randi([1, n], 1, n_ids);
end

% fprintf('using %f descriptors for learning Kmeans\n',n_ids);
for t = 1:T
    
    [C, dis, assign] = yael_kmeans (single(X), k);
    C = double(C);
    %fprintf(2,'\nreconstruction error Kmeans = %f\n\n',sum(double(dis)));
    
%     t0 = cputime;
    
    %% Note: To make initialization faster on large-scale dataset, we just 
    %  pick a small amount of samples for evaluation
     [ gama_all, f_total, f_total_locality, f_total_reconstruction] = ...
                                                learn_coeff_all_sample(X(:, ids) , C , mu);
%      fprintf('f_total = %f             f_total_locality = %f            f_total_recontruction = %f\n\n',...
%                                            f_total, f_total_locality, f_total_reconstruction);
      
      [ f_total, f_total_locality, f_total_reconstruction] = get_objective(X(:, ids), C, gama_all, mu);
%       fprintf('f_total = %f             f_total_locality = %f            f_total_recontruction = %f\n\n', ...
%                                                     f_total, f_total_locality, f_total_reconstruction);
            
%      fprintf('ITERATION %d: Elapsed time when learning gama_all is %f\n', t,  cputime - t0);
       
    if f_total < f_total_best
      f_total_best = f_total;
      f_total_locality_best = f_total_locality;
      f_total_reconstruction_best = f_total_reconstruction;
      gama_all_best = gama_all;
      B_best = C;
    end
     
end

stat.mu = mu;
stat.B_init = B_best;
stat.B = B_best;
stat.gama_all = gama_all_best;
stat.f_total = f_total_best;
stat.f_total_locality = f_total_locality_best;
stat.f_total_reconstruction = f_total_reconstruction_best;

% fprintf(2,'f_total_best with Kmeans = %f\n\n',f_total_best);


