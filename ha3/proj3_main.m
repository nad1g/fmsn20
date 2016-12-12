%
% Home Assignment 3
%

im = imread('titan.jpg'); im = double(im)/255;
imsz = size(im);
N = numel(im);
[u1,u2] = ndgrid(1:imsz(1), 1:imsz(2));
U = [u1(:),u2(:)];

NITER=1000;
k = zeros(NITER,1);
tau = zeros(NITER,1);
pc = zeros(NITER,1);
sigma_eps = zeros(NITER,1);

% Y = observations
Y = im(:);
b = mean(Y);

% x = GMRF
x = Y-b;

% Generate Q_sar
kappa = 2;
tau(1) = rand();
[C,G,G2] = matern_prec_matrices(U);
Q_0 = (kappa^4*C + 2*kappa^2 + G2);
Q = tau(1)*Q_0;

% Initialize pc, z
pc(1) = rand();
z = rand(imsz) > pc(1);
ab = 2; bb = 3; % hyper-parameters for the beta distribution

% Initialize sigma_eps
sigma_eps(1) = rand();

% sample conditional posteriors...
for ii = 2:NITER,
    disp(['iteration ',num2str(ii)]);
    % sample tau
    alpha_g = (N/2)+1; 
    beta_g = x'*Q_0*x/2;
    tau(ii) = gamrnd(alpha_g,1/beta_g);
    
    % sample sigma_eps
    known_idx = find(~z);
    n = length(known_idx);
    alpha_ig = (n/2)-1;
    beta_ig = sum((Y(known_idx)-(x(known_idx)+b)).^2)/2;
    sigma_eps(ii) = 1/gamrnd(alpha_ig, 1/beta_ig);

    % sample pc
    k(ii) = binornd(N,pc(ii-1));
    pc(ii) = betarnd(k(ii)+ab, N-k(ii)+bb);
    
    % determine p(z), classify..
    thresh = (2*sigma_eps(ii)^2) - log(pc(ii)/(2*pi*sigma_eps(ii)^2*(1-pc(ii))));
    z = (Y - (x + b*ones(size(x)))).^2 < thresh;
    
    % sample x
    A = sparse(1:length(Y(known_idx)),known_idx, 1, length(Y(known_idx)), N);
    Qbeta = speye(1)*1e-6;
    Q = tau(ii)*Q_0; Qall = blkdiag(Q,Qbeta);
    Qxy = Q + A'*A/sigma_eps(ii)^2;
    p = amd(Qxy);
    Qxy = Qxy(p,p);
    R = chol(Qxy);
    A = A(:,p);
    x = b + R\randn(size(R,1),1);
    x(p) = x;
    Exy = Qxy\(Aall'*Qeps*Y);
    Exy(p) = Exy;
    b = x(end);
end