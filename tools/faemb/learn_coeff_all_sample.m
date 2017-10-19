function [ gama_all, f_total, f_total_locality, f_total_reconstruction] = learn_coeff_all_sample(X, V, mu, stage)
%LEARN_COEFF_ALL_SAMPLE Summary of this function goes here
% compute coefficient of samples using closed form
% input: 
% X: set of samples
% V: bases (centroids)

if (~exist('stage', 'var'))
    stage = 'train';
end
f_total = 0;
f_total_locality = 0; 
f_total_reconstruction = 0;

m = size(X,2); %number of samples
n = size(V,2); %number of bases

gama_all = zeros(n,m);

%learn coefficient for one sample
for i = 1:m
    xi = X(:,i);
    
    %compute gama using closed form
    switch stage
        case 'train'
            gama_all(:,i) = learn_coeff_one_sample(xi, V, mu);
              % for checking.
%             [ f,   f_locality, f_reconstruction ] = get_objective( xi, V, gama, mu, emb_method );
%             f_total_locality = f_total_locality +f_locality;
%             f_total_reconstruction = f_total_reconstruction + f_reconstruction;
%             f_total = f_total + f;
        case 'test'
            gama_all(:,i) = learn_coeff_one_sample(xi, V, mu);
    end    
end
%% [ f_total,   f_total_locality, f_total_reconstruction ] = get_objective( X, V, gama_all, mu, emb_method );

