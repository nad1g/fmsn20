function x = gmrf_sample (mu,Q,M)
% x = gmrf_sample(mu,Q) samples from a Normal distribution
% mu = mean
% Q = precison matrix
% M = number of samples (M>1)
% x = sampled field

p = amd(Q);
R = chol(Q(p,p));
mu = mu(p);
sz = [size(Q,1) M];
x = mu + R\randn(sz);
x(p) = x;
