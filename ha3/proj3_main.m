%
% Home Assignment 3
%

im = imread('titan.jpg'); im = double(im)/255;
nn = rand(size(im)) < .25; nidx = find(nn == 1);
imsz = size(im);
N = numel(im);


NITER=1000;
k = zeros(NITER,1);
tau = zeros(NITER,1);
pc = zeros(NITER,1);
sigma_eps = zeros(NITER,1);
mu = zeros(NITER,1);

% Y = observations
Y = im(:) + 0.01*randn(N,1);
Y(nidx) = rand(size(nidx));
b = mean(Y);

% A matrix for image
Aimg = [speye(N) ones(N,1)];

% x = GMRF
x = Y-b;
x(end+1) = 0;

% Generate Q_sar
kappa = 0.05;
tau(1) = rand();
q = [0 -1 0;-1 4+kappa^2 -1;0 -1 0];
Q_0 = gmrfprec(imsz,q);
%[u1,u2] = ndgrid(1:imsz(1),1:imsz(2));
%[C,G,G2] = matern_prec_matrices([u1(:),u2(:)]);
%Q_0 = kappa^4*C + 2*kappa^2*G + G2;

% Initialize pc, z
pc(1) = .55;
k(1) = round(N*pc(1));

z = rand(N,1) > pc(1);
ab = 1; bb = 1; % hyper-parameters for the beta distribution

% Initialize sigma_eps
sigma_eps(1) = 1;

% sample conditional posteriors...
for ii = 2:NITER,
    disp(['iteration ',num2str(ii)]);
    
    good_idx = find(z==0);

    if isempty(good_idx)
       disp('all z set to 1');
       break;
    end

    A = sparse(1:length(Y(good_idx)),good_idx, 1, length(Y(good_idx)), N);
    Aall = [A ones(length(Y(good_idx)),1)]; %this is A_tilde

    % sample tau
    alpha_g = (N/2)+1; 
    beta_g = x(1:end-1)'*Q_0*x(1:end-1)/2;
    tau(ii) = gamrnd(alpha_g,1/beta_g);
    
    % sample sigma_eps^2
    n = length(good_idx);
    alpha_ig = (n/2)-1;
    beta_ig = sum((Y(good_idx)-Aall*x).^2)/2;
    sigma_eps(ii) = 1/gamrnd(alpha_ig, 1/beta_ig);

    % sample pc
    %k(ii) = binornd(N,pc(ii-1));
    k(ii) = sum(z == 0);
    pc(ii) = betarnd(k(ii)+ab, N-k(ii)+bb);
    
    % determine p(z), classify..
    pzk0 = 1/sqrt(2*pi*sigma_eps(ii-1)) * exp ((Y-Aimg*x).^2/(2*sigma_eps(ii-1)));
    pzk1 = ones(N,1);
    pz = (pzk0 * pc(ii))./(pzk0 * pc(ii) + pzk1 * (1-pc(ii)));
    z = rand(size(pz)) > pz;

    % sample x
    Qbeta = 1e-6*speye(1);
    Q = tau(ii-1)*Q_0; Qall = blkdiag(Q,Qbeta);
    Qeps = 1/(sigma_eps(ii-1)) * speye(length(Y(good_idx)));
    Qxy = Qall + Aall'*Qeps*Aall;

    p = amd(Qxy);
    Qxy = Qxy(p,p);
    R = chol(Qxy);
    Aallp = Aall(:,p);	

    % compute Exy
    Exy = Qxy\(Aallp'*Qeps*Y(good_idx));

    % sample x
    x = Exy + R\randn(size(R,1),1);

    % re-arrange
    x(p) = x;
    Exy(p) = Exy;

    mu(ii) = x(end);

end

Ezy = [speye(size(Q_0,1)) ones(size(Q_0,1),1)]*Exy;
figure, 
subplot(221), imagesc(im); colormap(gray); title('Original Image');
subplot(222), imagesc(reshape(Y,imsz)); colormap(gray); title('Original Img + Noise');
subplot(223), imagesc(reshape(Ezy,imsz)); colormap (gray); title(['Reconstructed Image (\kappa = ',num2str(kappa),')']);
subplot(224), imagesc(reshape(abs(Y-Ezy), imsz)); colormap(gray); colorbar; title('Obs - Reconstr');
figure,
subplot(221), plot(mu), title('\beta'); axis tight;
subplot(222), plot(tau), title('\tau'); axis tight;
subplot(223), plot(k/N), title('p_c'); axis tight;
subplot(224), semilogy(sigma_eps), title('\sigma \epsilon^2'); axis tight;
