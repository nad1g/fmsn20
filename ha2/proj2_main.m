%
% Main script
%

%load data
load HA2_Parana.mat
Y = rainfall;
Nobs = size(mesh.loc_obs,1);
Nmesh = size(mesh.loc,1);

%matrix of covariates for triangulation and observations
Bobs = [ones(Nobs,1) mesh.loc_obs(:,1) mesh.dist_obs mesh.elevation_obs];
Bmesh = [ones(Nmesh,1) mesh.loc(:,1) mesh.dist mesh.elevation];

%compute precision matrices for Matern-fields
[C, G, G2] = matern_prec_matrices(mesh.loc, mesh.T);

%create the A_tilde matrices
ABobs = [mesh.A Bobs];
ABall = [speye(size(mesh.loc,1)) Bmesh];

%Pick a validation and an observation set [validation = 10% of observation]
Iobs = rand(Nobs,1)>.1;
Ivalid = ~Iobs;

%we need a global variable for x_mode to reuse it
%between optimisation calls
global x_mode;

%attempt to estimate parameters (optim in optim...)
theta0 = rand(3,1);
theta = fminsearch( @(x) gmrf_negloglike_Gam(x, Y(Iobs), ABobs(Iobs,:), C, G, G2, 2), theta0);
disp('theta')
theta
disp('exp(theta)')
exp(theta)
H = gmrf_param_hessian(@(th) gmrf_negloglike_Gam(th,Y(Iobs),ABobs(Iobs,:),C, G, G2, 2), theta);
Hinv = inv(H);
th_se = sqrt(diag(Hinv));
disp('SE(exp(theta))');
exp(th_se)

%use the taylor expansion to compute posterior precision
%extract parameters
tau = exp(theta(1));
kappa2 = exp(theta(2));
b = exp(theta(3));
Q_x = tau*(kappa2^2*C + 2*kappa2*G + G2);
%combine this Q and Qbeta prior for regression coefficients
Qbeta = 1e-3 * speye(size(ABobs,2)-size(Q_x,2));
Qall = blkdiag(Q_x, Qbeta);
[~, ~, Q_xy] = gmrf_taylor_Gam(x_mode, Y(Iobs), ABobs(Iobs,:), Qall, b);


%ordinary regression (OLS)
[beta_,resid,sigma2,Sigma]= ols(Bobs(Iobs,:),log(Y(Iobs)));

for ii = 1:length(beta_)
   fprintf('beta(%d) = %1.4f\t SE=%1.4f\n', ii,beta_(ii), sqrt(Sigma(ii,ii)));
end
Y_ols = exp(Bobs(Ivalid,:)*beta_);
V_Y_ols = exp(sum((Bobs(Ivalid,:)*Sigma).*Bobs(Ivalid,:),2));

figure, plot(Y(Ivalid),'k-*');
hold on;
plot(Y_ols,'r--o');
plot(Y_ols - 1.96*sqrt(V_Y_ols),'b--');
plot(Y_ols + 1.96*sqrt(V_Y_ols),'b--');
xlabel('Location')
ylabel('Estimated precipitation')
title('OLS')

%compare it to GMRF
%for conf intervals, sample from Q_xy
R_xy = chol(Q_xy);
z = randn(size(R_xy,1),100);
v = R_xy\z;
Y_gmrf = exp(ABobs*(repmat(x_mode,1,100)+v));
Y_gmrf_mean = mean(Y_gmrf,2);
Y_gmrf_se = std(Y_gmrf,0,2);

figure, plot(Y(Ivalid),'k-*');
hold on;
plot(Y_gmrf_mean(Ivalid),'r--o');
plot(Y_gmrf_mean(Ivalid) - 1.96*Y_gmrf_se(Ivalid),'b--');
plot(Y_gmrf_mean(Ivalid) + 1.96*Y_gmrf_se(Ivalid),'b--');
xlabel('Location')
ylabel('Estimated precipitation')
title('GMRF')

lgp = ABall*x_mode;

figure,
trisurf(mesh.T, mesh.loc(:,1), mesh.loc(:,2),  ...
        zeros(size(mesh.loc,1),1), lgp);
hold on
plot(Border(:,1),Border(:,2),'-',...
  Border(1034:1078,1),Border(1034:1078,2),'-')
view(0,90); shading interp; colorbar;
hold off; axis tight

%calculate mean precipitation for mesh locations
Y_mesh_gmrf = exp(ABall*(repmat(x_mode,1,100)+v));
Y_mesh_gmrf_mean = mean(Y_mesh_gmrf,2);
Y_mesh_gmrf_se = std(Y_mesh_gmrf,0,2);

%plot the mesh predictions and standard error.
figure,
trisurf(mesh.T, mesh.loc(:,1), mesh.loc(:,2),  ...
        zeros(size(mesh.loc,1),1), Y_mesh_gmrf_mean);
hold on
plot(Border(:,1),Border(:,2),'-',...
  Border(1034:1078,1),Border(1034:1078,2),'-')
view(0,90); shading interp; colorbar;
hold off; axis tight
title('GMRF (mean)');

figure,
trisurf(mesh.T, mesh.loc(:,1), mesh.loc(:,2),  ...
        zeros(size(mesh.loc,1),1), Y_mesh_gmrf_se);
hold on
plot(Border(:,1),Border(:,2),'-',...
  Border(1034:1078,1),Border(1034:1078,2),'-')
view(0,90); shading interp; colorbar;
hold off; axis tight
title('GMRF (SE)');

%simulate x_mode from p(x|y,theta) (Gaussian approx) N(0,Q_xy^-1)
x_mode_sim = repmat(x_mode,1,100) + v;

%covariates
b1 = x_mode_sim(end-4,:);
b1 = x_mode_sim(end-3,:);
b2 = x_mode_sim(end-2,:);
b3 = x_mode_sim(end-1,:);
b4 = x_mode_sim(end,:);
figure
subplot(221)
hist(b1)
subplot(222)
hist(b2)
subplot(223)
hist(b3)
subplot(224)
hist(b4)

zz = exp(ABall*x_mode_sim);
sim_y = gamrnd(b,zz/b);
var_sim_y = var(sim_y,0,2);
median_sim_y = median(sim_y,2);


figure,
trisurf(mesh.T, mesh.loc(:,1), mesh.loc(:,2),  ...
        zeros(size(mesh.loc,1),1), median_sim_y);
hold on
plot(Border(:,1),Border(:,2),'-',...
  Border(1034:1078,1),Border(1034:1078,2),'-')
view(0,90); shading interp; colorbar;
hold off; axis tight
title('Median prediction');

figure,
trisurf(mesh.T, mesh.loc(:,1), mesh.loc(:,2),  ...
        zeros(size(mesh.loc,1),1), sqrt(var_sim_y));
hold on
plot(Border(:,1),Border(:,2),'-',...
  Border(1034:1078,1),Border(1034:1078,2),'-')
view(0,90); shading interp; colorbar;
hold off; axis tight
title('SE of the prediction');