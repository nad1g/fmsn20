%
% Universal Kriging
%

load HA1_SE_Temp
sz = [273 260];
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


for iter=1:3
  % compute binned covariance estimate
  [rhat,s2hat,m,n,d] = covest_nonparametric(U,resid,Kmax,Dmax);
  % use covest_ls to estimate covariance parameters
  par = covest_ls(rhat(1:6),s2hat,m,n,d(1:6),par_fixed);
  % estimate new 'beta's.
  sigma2 = par(1); kappa = par(2); nu = par(3);
  Sigma = matern_covariance(D,sigma2,kappa,nu);
  % account for the nugget variance (sigma_eps^2 = s2hat)
  Sigma = Sigma + s2hat * eye(size(Sigma)); 
  Sigma_inv_A = Sigma\A; Sigma_inv_y = Sigma\y;
  beta_ = (A'*Sigma_inv_A)\(A'*Sigma_inv_y);
  % compute residuals
  resid = y-A*beta_;
end

% co-ordinates of the 'validation' locations
U_uu = [SweObs(I_val,1) SweObs(I_val,2)];
D_uu = distance_matrix(U_uu);
Sigma_uu = matern_covariance(D_uu, sigma2, kappa, nu);

D_uk = distance_matrix(U_uu, U);
Sigma_uk = matern_covariance(D_uk, sigma2, kappa, nu);

% estimate for validation locations
% Yu = (A_u*beta_) + Sigma_uk * inv(Sigma) * resid
A_u = [ones(length(I_val),1) SweObs(I_val,[2,3,4,5])];
y_val = (A_u * beta_) + Sigma_uk * (Sigma \ resid);
% calculate confidence intervals.
vv = Sigma_uu - Sigma_uk * (Sigma \ Sigma_uk');
var_y_val = diag(vv);
se_y_val = sqrt(var_y_val);
ci_low = y_val - 1.96*se_y_val;
ci_hi = y_val + 1.96*se_y_val;
y_true = SweObs(I_val,6);
figure, plot(y_true,'k--');
hold on
plot(y_val,'r-o')
figure, plot(SweObs(I_val,6),'k-x');
hold on
plot(y_val,'r-o');
plot(ci_low,'b--');
plot(ci_hi,'b--');
xlabel('Validation locations');
ylabel('Avg. Temp.');
title('Predictions (universal Kriging)')
legend('True','Predicted')

% predictions for swegrid
long = reshape(SweGrid(:,1), sz);
Ind = ~isnan(long);
grid = SweGrid(Ind,:);
mu = nan(sz);
% The regressor matrix A for the SweGrid unknown locations
A_grid = [ones(length(grid),1) grid(:,[2,3,4,5])];
% Co-ordinate matrix for the unknown locations
U_uu = [grid(:,1) grid(:,2)];
% Calculate Sigma_uk (cross-cov between unknown, known locs)
D_uk = distance_matrix(U_uu, U);
Sigma_uk = matern_covariance(D_uk, sigma2, kappa, nu);
% predict y for the grid (y_grid)
y_grid = (A_grid * beta_) + Sigma_uk * (Sigma \ resid);
% save it to plot it!
mu(Ind) = y_grid;
% NOTE: do not use the variance formula directly. We just need the diagonal
% elements. So,
% First term is diag(Sigma_uu). However, note that diag(D_uu) is just 0s.
% Thus diag(Sigma_uu) = matern_covariance([0......0]', pars)
D_diag_uu = zeros(length(grid),1);
Sigma_diag_uu = matern_covariance(D_diag_uu, sigma2, kappa, nu);
% Second term is diag(Sigma_uk * inv(Sigma) * Sigma_ku)
% but Sigma_ku = Sigma_uk' (due to symmetry).
term2 = Sigma_uk * (Sigma \ Sigma_uk');
var_y_grid = Sigma_diag_uu - diag(term2);
figure,
imagesc([11.15 24.15], [69 55.4], mu, 'alphadata', Ind)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off; colorbar
title('Predictions (Kriging)')
se = nan(sz);
se(Ind) = sqrt(var_y_grid);
figure,
imagesc([11.15 24.15], [69 55.4], se, 'alphadata', Ind)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off; colorbar;
title('Standard Error of prediction (Kriging)');
