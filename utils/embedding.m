function xe = embedding(x, param, encmethod)

xe = yael_vecs_normalize (param.Ud' * bsxfun (@minus, x, param.desc_mean), 2, 0);

% Encoding method
switch encmethod
    case 'temb'
        xe = param.Pemb * triemb_sumagg (xe, param.C, param.Xmean);
    case 'tembsink'
        xe = qdemocratic (yael_vecs_normalize(triemb_map (xe, param.C, param.Pemb, param.Xmean)), 'sinkhorn');
    case 'faemb'
        [ gamma_i ] = learn_coeff_all_sample(xe, param.B , param.mu, 'test');
        xe = single(fa_embedding(xe, single(param.B), single(gamma_i))); 
        xe = qdemocratic( yael_vecs_normalize(param.Pemb * bsxfun(@minus, xe, param.Xmean), 2, 0), 'sinkhorn', 0.5);
end
