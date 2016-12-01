%
% Main script
%

%load data
load HA2_Parana.mat
Y = rainfall;
Nobs = size(mesh.loc_obs,1);

%matrix of covariates for triangulation and observations
Bobs = [ones(Nobs,1) mesh.loc_obs(:,1) mesh.dist_obs mesh.elevation_obs];
%Bmesh = ?

%compute precision matrices for Matern-fields
[C, G, G2] = matern_prec_matrices(mesh.loc, mesh.T);

%create the A_tilde matrices
ABobs = [mesh.A Bobs];
%ABall = [speye(size(mesh.loc,1)) Bmesh];

%Pick a validation and an observation set [validation = 10% of observation]
Iobs = rand(Nobs,1)>.1;
Ivalid = ~Iobs;

%we need a global variable for x_mode to reuse it
%between optimisation calls
global x_mode;

%attempt to estimate parameters (optim in optim...)
theta0 = rand(3,1);
theta = fminsearch( @(x) gmrf_negloglike_Gam(x, Y(Iobs), ABobs(Iobs,:), C, G, G2, 2), theta0);

%use the taylor expansion to compute posterior precision
[~, ~, Q_xy] = gmrf_taylor_Gam(x_mode, Y(Iobs), AB(Iobs), Qall, b);

e = sparse(size(spde.C,1)+1, 1, 1, size(Q_xy,1),1);
V_beta0 = e'*(Q_xy\e);
