% Load data
load fmri.mat

%size of data
sz = size(img);

%Option 1: regress onto indicator functions
beta = X\colstack(img)';
beta = icolstack(beta',sz(1:2));
%and treat the beta:s as the image
[y_beta, ~, P_beta] = pca(colstack(beta));
y_beta = icolstack(y_beta, sz(1:2));

figure
subplot(3,4,1)
semilogy(P_beta/sum(P_beta))
axis tight
for i=1:size(y_beta,3)
  subplot(3,4,i+1)
  imagesc(y_beta(:,:,i))
  title(i)
end

%Option 2: Compute SVD directly on the data
[y,V,P] = pca(colstack(img));
y = icolstack(y, sz(1:2));

%study the temporal components to find those with 20s periodicity
figure
subplot(3,4,1)
semilogy(P/sum(P))
axis tight
for i=1:11
  subplot(3,4,i+1)
  plot(V(:,i))
  axis tight
  title(i)
end

figure
subplot(3,4,1)
semilogy(P/sum(P))
axis tight
for i=1:11
  subplot(3,4,i+1)
  imagesc(y(:,:,i))
  title(i)
end
