% estimate kappa^2

% read image and plot
im = imread('titan.jpg'); im = double(im)/255;
figure, subplot(121), imagesc(im); colormap(gray);

imsz = size(im);
[u1,u2] = ndgrid(1:imsz(1),1:imsz(2));

%subsample the image
idx = rand(imsz) > 0.75;
Y = im(idx); u1 = u1(idx); u2 = u2(idx);
Y = Y(:);
x = Y - mean(Y);
U = [u1(:),u2(:)];
D = distance_matrix(U);

Dmax = max(max(D))+0.1;
Kmax = 30;

% non-parametric covariance estimation...
[rhat,s2hat,m,n,d] = covest_nonparametric([u1(:),u2(:)],x,Kmax,Dmax);
figure, plot(d, rhat, 'k-x'); hold on;
N = length(x);
% bootstrap
rr = zeros(50,length(rhat));
for ii = 1:50,
   idx = randperm(N);
   xp = x(idx); % randomly permuted residuals.
   [rr(ii,:),s2hat,m,n,d] = covest_nonparametric([u1(:),u2(:)],xp,Kmax,Dmax);
end
 
r_quant = quantile(rr,[.25 .75],1);
plot(d,r_quant(1,:),'b--');
plot(d,r_quant(2,:),'b--');

NITER = 10;
par_fixed = zeros(4,1);
A = ones(length(Y),1);
for iter = 1:NITER,
    [rhat,s2hat,m,n,d] = covest_nonparametric([u1(:),u2(:)],x,Kmax,Dmax);
    par = covest_ls(rhat(1:9),s2hat,m,n,d(1:9),par_fixed);
    sigma2 = par(1); kappa = par(2); nu = par(3);
    Sigma = matern_covariance(D,sigma2,kappa,nu);
    %add nugget
    Sigma = Sigma + sigma2 * eye(size(Sigma));
    Sigma_Inv_Y = Sigma\Y;
    Sigma_Inv_A = Sigma\A;
    b = (A'*Sigma_Inv_A)\(A'*Sigma_Inv_Y);
    x = Y-A*b;
end