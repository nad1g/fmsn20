%
% check the Taylor series expansion of the conditional
%

%load data
load HA2_Parana.mat
B = [ones(length(mesh.loc_obs),1) mesh.loc_obs(:,1) mesh.dist_obs mesh.elevation_obs];
AB = [mesh.A B];
y = rainfall;

%compute precision matrices for Matern-fields
[spde.C, spde.G, spde.G2] = matern_prec_matrices(mesh.loc, mesh.T);

%set the parameters
tau = 0.997;
kappa2 = 1.002;
b = 10.12;

%compute Q for a SAR process
Q_x = tau*(kappa2^2*spde.C + 2*kappa2*spde.G + spde.G2);

%combine this Q and Qbeta prior for regression coefficients
Qbeta = 1e-3 * speye(size(AB,2)-size(Q_x,2));
Qall = blkdiag(Q_x, Qbeta);

%declare x_mode as global so that we start subsequent optimisations from
%the previous mode (speeds up nested optimisation).
global x_mode;
if isempty(x_mode)
  %no existing mode, compute a rough initial guess assuming Gaussian errors
  x_mode = (Qall + AB'*AB)\(AB'*log(y+.1));
end

%figure, plot(x_mode); hold on;
%nested optimisation to find x_mode using Newton-Raphson optimisation
x_mode = fminNR(@(x) gmrf_taylor_Gam(x, y, AB, Qall, b), x_mode);

y_rec = exp(AB*x_mode);
figure, plot(y,'bo'); hold on; plot(y_rec,'r-.');
%plot(x_mode,'r');
