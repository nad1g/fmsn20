%
% Home Assignment 3
%

% read image and plot
im = imread('titan.jpg'); im = double(im);
figure, subplot(121), imagesc(im); colormap(gray);

% decrease resolution
y = im(1:4:end,1:4:end);
subplot(122), imagesc(y), colormap(gray);
[m,n] = size(y);
[u1,u2] = ndgrid(1:m,1:n);
D = distance_matrix([u1(:), u2(:)]);
Dmax = max(D(:))+0.1;

%compute mean component by regressing with co-ordinates
y = colstack(y);
r = y - mean(y);

par_fixed = zeros(4,1); par_fixed(3) = 5;
Nr = length(r);
Kmax = 20;
U = [u1(:),u2(:)];
% non-parametric covariance estimation...
[rhat,s2hat,m,n,d] = covest_nonparametric(U,r,Kmax,Dmax);

% build quantiles from 100 bootstrap

for ii = 1:100,
  idx = randperm(Nr);
  rp = r(idx); % randomly permuted residuals.
  [rr(ii,:),s2hat,m,n,d] = covest_nonparametric(U,rp,Kmax,Dmax);
end

% plot the covariance function, quantiles. see how many points stick out of the quantiles.
figure; boxplot(rr); hold on; plot(rhat,'k-*'); xlabel('Bin'); ylabel('r(h)');
A = ones(length(y),1);
NITER=4;
for iter = 1:NITER
  % compute binned covariance estimate
  [rhat,s2hat,m,n,d] = covest_nonparametric(U,r,Kmax,Dmax);
  % use covest_ls to estimate covariance parameters
  par = covest_ls(rhat(1:8),s2hat,m,n,d(1:8),par_fixed);
  % estimate new 'beta's.
  sigma2 = par(1); kappa = par(2); nu = par(3);
  Sigma_kk= matern_covariance(D,sigma2,kappa,nu);
  % account for the nugget variance (sigma_eps^2 = s2hat)
  Sigma_kk = Sigma_kk + s2hat * eye(size(Sigma_kk)); 
  Sigma_inv_A = Sigma_kk\A; Sigma_inv_y = Sigma_kk\y;
  b = (A'*Sigma_inv_A)\(A'*Sigma_inv_y);
  % compute residuals
  r = y-A*b;
end
% estimate a 'kappa' which shall be fixed.
%par = estimate_matern_param(y,par_init,show_plot);


