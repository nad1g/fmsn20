function  [x,res,sigma2,Sigma]= ols(A,y)
% [x,sigma2,Sigma]= ols(A,y) 
% Implements the ordinary least squares method for solving A*x = y.
% A = matrix containing covariates
% y = observations
% x = estimated parameters
% res = residuals, i.e., y - A*x
% sigma2 = estimated variance of the residuals
% Sigma = covariance matrix

y = y(:);
% number of observations N
N = length(y);
if (size(A,1) ~= N)
    error('Num columns of A not equal to length of y');
end
p = size(A,2);
R = A'*A;
z = A'*y;
% estimate the parameters
x = R\z;
% compute the residuals
res = y - A*x;
% sigma_e^2 (in lecture notes)
sigma2 = res'*res /(N-p);
% compute covariance matrix
Sigma = sigma2 * inv(R);