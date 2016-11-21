%
% Universal Kriging
%

%% Data
load HA1_SE_Temp
sz = [273 260];
lng = reshape(SweGrid(:,1), sz);
Ind = ~isnan(lng);
swe_grid = SweGrid(Ind,:);

I_val = randperm(250,25);
I_obs = 1:250; I_obs(I_val) = [];

y_k = SweObs(I_obs,6); % known observations
y_v = SweObs(I_val,6); % validation data (truth)

% Co-ordinates
Uk = [SweObs(I_obs,1) SweObs(I_obs,2)]; % co-ordinates of known data
Uv = [SweObs(I_val,1) SweObs(I_val,2)]; % co-ordinates of validation data
Uu = [swe_grid(:,1) swe_grid(:,2)]; % co-ordinates whos avg temp is not known (to be predicted)

% Distance Matrices
Dk = distance_matrix(Uk);     % known, known
Dvk = distance_matrix(Uv,Uk); % validation, known
Duk = distance_matrix(Uu,Uk); % unknown, known

% OLS model and residuals
A_k = [ones(length(I_obs),1) SweObs(I_obs,[2,3,4,5])];
[b,r,s2,S] = ols(A_k, y_k);

%% Non-parametric estimation of covariance function (binned estimate)
% parameters:
Kmax = 32; % num bins = Kmax+1
Dmax = max(Dk(:))+0.1;
par_fixed = zeros(4,1); %par_fixed(3) = 1.5; %parameters
Nr = length(r); %number of residuals

% non-parametric covariance estimation...
[rhat,s2hat,m,n,d] = covest_nonparametric(Uk,r,Kmax,Dmax);

% build quantiles from 100 bootstrap

for ii = 1:100,
  idx = randperm(Nr);
  rp = r(idx); % randomly permuted residuals.
  [rr(ii,:),s2hat,m,n,d] = covest_nonparametric(Uk,rp,Kmax,Dmax);
end

% plot the covariance function, quantiles. see how many points stick out of the quantiles.
figure; boxplot(rr); hold on; plot(rhat,'k-*'); xlabel('Bin'); ylabel('r(h)');

% also, plot the number of data points per bin (should ignore results from bins with few data points)
figure, plot(m,'ko'); xlabel('Bin'); ylabel('Num samples/bin');

%% Re-estimate mean
NITER=3;

for iter = 1:NITER
  % compute binned covariance estimate
  [rhat,s2hat,m,n,d] = covest_nonparametric(Uk,r,Kmax,Dmax);
  % use covest_ls to estimate covariance parameters
  par = covest_ls(rhat(1:6),s2hat,m,n,d(1:6),par_fixed);
  % estimate new 'beta's.
  sigma2 = par(1); kappa = par(2); nu = par(3);
  Sigma_kk= matern_covariance(Dk,sigma2,kappa,nu);
  % account for the nugget variance (sigma_eps^2 = s2hat)
  Sigma_kk = Sigma_kk + s2hat * eye(size(Sigma_kk)); 
  Sigma_inv_A = Sigma_kk\A_k; Sigma_inv_y = Sigma_kk\y_k;
  b = (A_k'*Sigma_inv_A)\(A_k'*Sigma_inv_y);
  % compute residuals
  r = y_k-A_k*b;
end

% we now have refined estimates of the parameters b. compute variance
vb = diag( s2hat * inv(A_k'*Sigma_inv_A) );

%% Validation
A_v = [ones(length(I_val),1) SweObs(I_val,[2,3,4,5])];
Sigma_vk = matern_covariance(Dvk,sigma2,kappa,nu);
s0 = matern_covariance(0,sigma2,kappa,nu);
[y_val_est, y_val_se] = kriging(A_k, A_v, b, Sigma_kk, Sigma_vk, r, s0);
% plot...
figure; hold on;
y_est_ci_low = y_val_est - 1.96*y_val_se; y_est_ci_hi = y_val_est + 1.96*y_val_se;
plot(y_v,'k--'); plot(y_val_est,'r-o'); 
plot(y_est_ci_low,'b--'); plot(y_est_ci_hi,'b--');
xlabel('Location'); ylabel('Temperature');
legend('Truth','Prediction');

%% Prediction
A_u = [ones(length(swe_grid),1) swe_grid(:,[2,3,4,5])];
Sigma_uk = matern_covariance(Duk, sigma2, kappa, nu);
s0 = matern_covariance(0,sigma2,kappa,nu);
[yu_est, yu_est_se] = kriging(A_k, A_u, b, Sigma_kk, Sigma_uk, r, s0);
% plot...
mu = nan(sz);
se = nan(sz);
mu(Ind) = yu_est; se(Ind) = yu_est_se;
figure,
imagesc([11.15 24.15], [69 55.4], mu, 'alphadata', Ind)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off; colorbar
title('Predictions (Kriging)')
figure,
imagesc([11.15 24.15], [69 55.4], se, 'alphadata', Ind)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off; colorbar;
title('Standard Error of prediction (Kriging)');