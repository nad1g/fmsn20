function [yu_hat, yu_hat_se] = kriging(A_k,A_u,b,Sigma_kk, Sigma_uk,r,s0)
% [yu_hat, yu_hat_se] = kriging(A,b,Sigma_kk, Sigma_uk,r)
% implements the Kriging estimator.
% yu_hat = predictions
% yu_hat_se = standard error of the predictions
% A = covariate matrix
% b = parameters
% Sigma_xx are the covariance matrices as shown below
% | <-- k -->   <-u->
% k Sigma_kk, Sigma_ku
% |
% u Sigma_uk, Sigma_uu
% |
% s0 = covariance for h=0
% r = residuals (y_k - A_k*b)
yu_hat = A_u*b + Sigma_uk*(Sigma_kk\r);
G = A_u' - A_k'*(Sigma_kk\Sigma_uk');
H = inv(A_k'*(Sigma_kk\A_k));
yu_hat_var = s0 - sum((Sigma_uk*inv(Sigma_kk)).*Sigma_uk,2) + sum((G'*H).*G',2);
yu_hat_se = sqrt(yu_hat_var);