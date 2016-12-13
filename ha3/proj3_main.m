%
% Home Assignment 3
%

im = imread('titan.jpg'); im = double(im)/255;
nn = rand(size(im)) < .3; nidx = find(nn == 1);
imsz = size(im);
N = numel(im);
%[u1,u2] = ndgrid(1:imsz(1), 1:imsz(2));
%U = [u1(:),u2(:)];

NITER=500;
k = zeros(NITER,1);
tau = zeros(NITER,1);
pc = zeros(NITER,1);
sigma_eps = zeros(NITER,1);
mu = zeros(NITER,1);
% Y = observations
Y = im(:);
Y(nidx) = rand(size(nidx));
b = mean(Y);

% x = GMRF
x = Y-b;
x(end+1) = 0;
% Generate Q_sar
kappa = .05;
tau(1) = rand();
q = [0 -1 0;-1 4+kappa^2 -1;0 -1 0];
Q_0 = gmrfprec(imsz,q);
% Initialize pc, z
pc(1) = .5;
z = rand(N,1) > pc(1);
ab = 2; bb = 3; % hyper-parameters for the beta distribution

% Initialize sigma_eps
sigma_eps(1) = 1;

% sample conditional posteriors...
for ii = 2:NITER,
    disp(['iteration ',num2str(ii)]);
    % sample tau
    alpha_g = (N/2)+1; 
    beta_g = x(1:end-1)'*Q_0*x(1:end-1)/2;
    tau(ii) = gamrnd(alpha_g,1/beta_g);
    
    % sample sigma_eps
    known_idx = find(~z);
    n = length(known_idx);
    alpha_ig = (n/2)-1;
    beta_ig = sum((Y(known_idx)-x(known_idx)).^2)/2;
    sigma_eps(ii) = 1/gamrnd(alpha_ig, 1/beta_ig);
    if isnan(sigma_eps(ii))
        keyboard
    end

    % sample pc
    k(ii) = binornd(N,pc(ii-1));
    pc(ii) = betarnd(k(ii)+ab, N-k(ii)+bb);
    
    % determine p(z), classify..
    %thresh = (2*sigma_eps(ii)^2) - log(pc(ii)/(sqrt(2*pi*sigma_eps(ii)^2)*(1-pc(ii))));
    q = (1-pc(ii))/pc(ii);
    K = sqrt(2*pi*sigma_eps(ii)^2);
    al = ((Y - (x(1:end-1) + x(end))).^2)/(2*sigma_eps(ii)^2);
    pz = 1./(1 + K*q*exp(al)); % if pz(ii) > 0.5, it is 0, so...
    z = pz < 0.5;
    
    % sample x
    A = sparse(1:length(Y(known_idx)),known_idx, 1, length(Y(known_idx)), N);
    Aall = [A ones(length(Y(known_idx)),1)]; %this is A_tilde
    Qbeta = 1e-6*speye(1);
    Q = tau(ii)*Q_0; Qall = blkdiag(Q,Qbeta);
    Qeps = 1/(sigma_eps(ii)^2) * speye(length(Y(known_idx)));
    Qxy = Qall + Aall'*Qeps*Aall;
    p = amd(Qxy);
    Qxy = Qxy(p,p);
    R = chol(Qxy);
    Aall = Aall(:,p);
    x = b + R\randn(size(R,1),1);
    x(p) = x;
    %Exy = Qxy\(A'*Y(known_idx)/sigma_eps(ii)^2);
    Exy = Qxy\(Aall'*Qeps*Y(known_idx));
    Exy(p) = Exy;
    mu(ii) = x(end);
end

Ezy = [speye(size(Q_0,1)) ones(size(Q_0,1),1)]*Exy;
figure, imagesc(reshape(Ezy,imsz)); colormap (gray);
figure,
subplot(221), plot(mu), title('mu');
subplot(222), plot(tau), title('tau');
subplot(223), plot(pc), title('p_c');
subplot(224), plot(sigma_eps), title('sigma eps');