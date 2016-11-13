%
% lab 1
%

%%
sigma2 = 2;
nu = 4;
rho = 15;   % range, rho = sqrt(8*nu)/kappa
kappa = sqrt(8*nu)/rho;
dist = linspace(0,80,100);
r = matern_covariance(dist, sigma2, kappa, nu);
figure;
plot(dist,r);
xlabel('h');
ylabel('r(h)');
title(['\sigma^2 = ', num2str(sigma2), ' \rho = ', num2str(rho), ' \nu = ', num2str(nu)]);

%%
N = 3000;
sz = [50 60];
mu = 10;
% note: a 50x60 image has 3000 locations from (1,1)....(50,60)
[u,v] = ndgrid(1:50,1:60);
% [u(:),v(:)] are the co-ordinates (1,1)...(50,60)
D = distance_matrix([u(:),v(:)]);
% the covariance matrix Sigma is of size 3000x3000
Sigma = matern_covariance(D, sigma2, kappa, nu);
R = chol(Sigma);
eta = mu + R' * randn(N,1);
eta_image = reshape(eta,sz);
figure, imagesc(eta_image);
title('Simulated Field');

%%
sigma_eps = .1;
y = eta + randn(N,1)*sigma_eps;
z = y-mu;
figure, plot(D,z*z', '.k');
hold on;
plot(D,Sigma,'r.');
title('Covariance Cloud');

%%
% alt 2 (faster)
U = [u(:),v(:)]; % the 3000x2 co-ordinate matrix
Kmax = 50;
Dmax = max(D(:))+0.001;
[rhat,s2hat,m,n,d] = covest_nonparametric(U,z,Kmax,Dmax);
figure, plot(d,rhat,'o');
hold on;
plot(s2hat,'or');
r_theo = matern_covariance(d, sigma2, kappa, nu);
plot(d,r_theo,'r');

%%
par_fixed = zeros(4,1);
par_fixed(3) = 4;
par = covest_ls(rhat,s2hat,m,n,d,par_fixed);
disp('Param: True <> Observed');
disp(['\sigma^2: ',num2str(sigma2), ' <> ', num2str(par(1))]);
disp(['\kappa: ',num2str(kappa), ' <> ', num2str(par(2))]);
disp(['\nu: ',num2str(nu), ' <> ', num2str(par(3))]);
disp(['\sigma2_eps: ',num2str(sigma_eps^2), ' <> ', num2str(par(4))]);