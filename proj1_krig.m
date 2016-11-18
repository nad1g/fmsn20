load HA1_SE_Temp
sz = [273 260];
%
% OLS Model
%
%I_val = datasample(1:250,25,'Replace',false);
I_val = randperm(250,25);
I_obs = 1:250; I_obs(I_val) = [];

y = SweObs(I_obs,6);
A = [ones(length(I_obs),1) SweObs(I_obs,[2,3,4,5])];
[beta_,resid,sigma2,Sigma]= ols(A,y);

% we have the residuals now. C(y_i,y_j) = E[r_i,r_j]
U = [SweObs(I_obs,1) SweObs(I_obs,2)]; % co-ordinate matrix
D = distance_matrix(U);
cov_samples = resid*resid';
figure, plot(D,cov_samples,'k.');

% parameters for non-parametric cov est.
Kmax = 32;
Dmax = max(D(:))+0.1;
par_fixed = zeros(4,1);
par_fixed(3) = 4;
num_res = length(resid);
[rhat,s2hat,m,n,d] = covest_nonparametric(U,resid,Kmax,Dmax);
for iter = 1:100,
  p_idx = randperm(num_res);
  resid_p = resid(p_idx);
  [rr(iter,:),s2hat,m,n,d] = covest_nonparametric(U,resid_p,Kmax,Dmax);
end
boxplot(rr);
hold on;
plot(rhat,'k-*');


for iter=1:1
  % compute binned covariance estimate
  [rhat,s2hat,m,n,d] = covest_nonparametric(U,resid,Kmax,Dmax);
  % use covest_ls to estimate covariance parameters
  par = covest_ls(rhat(1:15),s2hat,m,n,d(1:15),par_fixed);
  % estimate new 'beta's.
  sigma2 = par(1); kappa = par(2); nu = par(3);
  % bit unsure here...
  Sigma = matern_covariance(D,sigma2,kappa,nu) + s2hat * eye(size(Sigma));
  Sigma_inv_A = Sigma\A; Sigma_inv_y = Sigma\y;
  beta_ = (A'*Sigma_inv_A)\(A'*Sigma_inv_y);
  % compute residuals
  resid = y-A*beta_;
end

% asses uncertainty in cov est.


