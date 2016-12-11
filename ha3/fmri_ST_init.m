% Load data
load fmri.mat

%X contains a constant and linear trend to model slow motion of the subject
%as well as indicators for the time-periods of study. In a real example
%we'd use one beta field for each stimulation period. However that model is
%VERY computationally expensive so to illustrate the principal we'll
%sum the indicators to obtain one indicator for an event during any of the
%time periods.
Xa = [X(:,1:2) sum(X(:,3:end),2)];

%illustrate the function in X and Xa
figure
subplot(211)
plot(X)
subplot(212)
plot(Xa)

%size of the data
sz = size(img);
nX = size(Xa,2);

%stack all the data into vector
Y = img(:);
%observation matrix linking beta to Y
A = kron(Xa, speye(sz(1)*sz(2)));
%independent regression estimates for each location can be obtained by
%regressing Y onto A
beta_OLS = A\Y;
%with residuals
e_OLS = Y-A*beta_OLS;
%beta_OLS and e_OLS now contains vectorised version of residuals and
%beta-estimates, lets reshape to the image size
beta_OLS = reshape(beta_OLS, sz(1), sz(2), nX);
e_OLS = reshape(e_OLS, sz(1), sz(2), sz(3));
%and compute s2 estimates for each pixel
s2_OLS = sum(e_OLS.^2,3) / (sz(3)-nX);
%For the uncertainty in beta we have that (using properties of kronecker
%products):
%  A'*A = kron(Xa,eye(N))'*kron(Xa,eye(N)) = kron(Xa'*Xa,eye(N)'*eye(N)) =
%       = kron(Xa'*Xa,eye(N))
%and
%  inv(A'*A) = inv( kron(Xa'*Xa,eye(N)) ) = kron(inv(Xa'*Xa),eye(N))
%Thus we only need to consider the diagonal elements of inv(Xa'*Xa) and
%specifically we only need the third element of we're looking at the
%uncertainty of the beta for the event indicator (column 3 of Xa)
iXX_33 = [0 0 1]*inv(Xa'*Xa)*[0;0;1];
%with different s2 estimates for each pixel the temporal trend is now
%significant if
%  abs(beta_OLS(:,:,3)) > lambda_alpha/2 * sqrt(s2_OLS*iXX_33)
%Here we have used that t-quantile is roughly a lambda quantile if f>100.
%To account for the fact that we're testing all the pixels we adjust the
%significans level using a Bonferoni correction so that alpha=alpha/N;
alpha = 0.025 / (sz(1)*sz(2));
I_significant = abs(beta_OLS(:,:,3)) > norminv(1-alpha)*sqrt(iXX_33*s2_OLS);

%Plots illustrating the results
figure
%estimated beta:s
for i=1:3
    subplot(2,3,i)
    imagesc(beta_OLS(:,:,i))
end
%estimated variances, s2
subplot(234)
imagesc(s2_OLS)
%significant beta_OLS_3 elements
subplot(235)
imagesc(I_significant)
%beta_OLS_3 field with significant pixels masked out
subplot(236)
imagesc(beta_OLS(:,:,3), 'alphadata', I_significant)