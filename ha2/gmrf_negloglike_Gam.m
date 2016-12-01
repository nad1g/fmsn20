function negloglike = gmrf_negloglike_Gam(theta, y, AB, C, G, G2, alpha)
% GMRF_NEGLOGLIKE_GAM Calculate the GMRF data likelihood, non-Gaussian observations
%
% negloglike = gmrf_negloglike_Gam(theta, y, AB, C, G, G2, alpha)
%
% theta - log([tau2; kappa; b])
% y - the data vector, as a column with n elements
% AB=[A B] - the observation matrix, sparse n-by-(N+Nbeta)
% C,G,G2 - matrices used to build a Mat�rn-like precision,
%          see matern_prec_matrices, sparse N-by-N
% alpha - Consider CAR (alpha=1) or SAR (alpha=2) field?
%
% This is only a skeleton for Home Assignment 2.

% $Id: gmrf_negloglike_skeleton.m 4454 2011-10-02 18:29:12Z johanl $


if nargin<7 || isempty(alpha), alpha=2; end

%extract parameters
tau = exp(theta(1));
kappa2 = exp(theta(2));
b = exp(theta(3));

%compute Q for a CAR or SAR process
if alpha==2
  Q_x = tau*(kappa2^2*C + 2*kappa2*G + G2);
end

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

%nested optimisation to find x_mode using Newton-Raphson optimisation
x_mode = fminNR(@(x) gmrf_taylor_Gam(x, y, AB, Qall, b), x_mode);

%find the Laplace approximation of the denominator
[f, ~, Q_xy] = gmrf_taylor_Gam(x_mode, y, AB, Qall, b);
%note that f = -log_obs + x_mode'*Q*x_mode/2.

%Compute choleskey factors
[R_x,p_x] = chol(Q_x);
[R_xy,p_xy] = chol(Q_xy);
if p_x~=0 || p_xy~=0
  %choleskey factor fail -> (almost) semidefinite matrix -> 
  %-> det(Q) ~ 0 -> log(det(Q)) ~ -inf -> negloglike ~ inf
  %Set negloglike to a REALLY big value
  negloglike = realmax;
  return;
end

%note that f = -log_obs + x_mode'*Q*x_mode/2.
%negloglike = f - x_mode'*Qall*x_mode/2 - sum(log(diag(R_x))) + sum(log(diag(R_xy)));
negloglike = f - sum(log(diag(R_x))) + sum(log(diag(R_xy)));
%print diagnostic information (progress)
fprintf(1, 'Theta: %11.4e %11.4e %11.4e; fval: %11.4e\n', ...
  theta(1), theta(2), theta(3), negloglike);
