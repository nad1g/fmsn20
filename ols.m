function  [beta_,Sigma_]= ols(D,y,c)
% [beta_,Sigma_]= ols(D,y,c_cov) 
% Implements the ordinary least squares method for f solving A*beta_ = y, 
% D = Matrix containing covariates
% y = observations
% beta_ = estimated parameters
% Sigma_ = variance of the estimated parameters
% c (optional) = vector containing the column numbers of D that
%                represent the covariates (default = all columns).

[N,~] = size(D);
y = y(:);
if length(y) ~= N
    error('length of obs not equal to num rows of regressor matrix.');
end

if nargin == 3,
    A = D(:,c);
end

if nargin == 2,
    A = D;
end

[beta_, Sigma_] = mvregress(A,y);