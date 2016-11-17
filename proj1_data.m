%set paths to course files and/or download HA1_SE_Temp.mat from the homepage

%% load data
load HA1_SE_Temp
%extract covariates and reshape to images
sz = [273 260];
long = reshape(SweGrid(:,1), sz);
lat = reshape(SweGrid(:,2), sz);
elev = reshape(SweGrid(:,3), sz);
dist = reshape(SweGrid(:,4), sz);
dist_swe = reshape(SweGrid(:,5), sz);
%points inside of Sweden (i.e. not nan)
Ind = ~isnan(long);

%% Plot the covariates
figure, subplot(3,2,1);
scatter(SweObs(:,1),SweObs(:,6),20,'filled');
xlabel('latitude');
subplot(3,2,2);
scatter(SweObs(:,2),SweObs(:,6),20,'r','filled');
xlabel('longitude');
subplot(3,2,3);
scatter(SweObs(:,3),SweObs(:,6),20,'g','filled');
xlabel('elevation');
subplot(3,2,4);
scatter(SweObs(:,4),SweObs(:,6),20,'m','filled');
xlabel('dist to any coast');
subplot(3,2,5);
scatter(SweObs(:,5),SweObs(:,6),20,'k','filled');
xlabel('dist to swe coast');
%subplot(3,2,6);
%scatter(SweObs(:,1),SweObs(:,6),20,'filled');

%% Plot data
%plot January observations
figure(1)
subplot(131)
scatter(SweObs(:,1), SweObs(:,2), 20, SweObs(:,6), 'filled')
axis xy tight; hold on
plot(Border(:,1),Border(:,2))
hold off; colorbar
title('June temperature')

%plot elevation (prediction surface + at observations sites)
subplot(132)
imagesc([11.15 24.15], [69 55.4], elev, 'alphadata', Ind)
axis xy; hold on
plot(Border(:,1),Border(:,2))
scatter(SweObs(:,1), SweObs(:,2), 25, SweObs(:,3),...
  'filled','markeredgecolor','k')
colorbar
hold off
title('Elevation')

%plot distance to coast (prediction surface + at observations sites)
subplot(133)
imagesc([11.15 24.15], [69 55.4], dist, 'alphadata', Ind)
axis xy; hold on
plot(Border(:,1), Border(:,2))
scatter(SweObs(:,1), SweObs(:,2), 25, SweObs(:,4),...
  'filled', 'markeredgecolor', 'k')
colorbar
hold off
title('Dist. Coast')

%% Example showing how to do predictions and fit them onto the grid
%pick out the relevant parts of the grid
grid = SweGrid(Ind,:);
%create a matrix holding "predicitons"
mu = nan(sz);
%do "predicitons"
N = size(SweObs,1);
A = [ones(N,1) SweObs(:,[2,3,5])];
y = SweObs(:,6);
[beta_,resid,sigma2,Sigma]= ols(A,y);
fprintf('-- Ordinary Least Squares --\n');
fprintf('Number of observations = %d\n', N);
fprintf('Variance of the residuals = %f\n', sigma2);
seBeta = sqrt(diag(Sigma));
for ii = 1:length(seBeta),
    fprintf('beta_%d = %f, SE (beta_%d) = %f\n', ii, beta_(ii), ii, seBeta(ii));
end
E = beta_(1) + grid(:,[2,3,5])*beta_(2:end);
%and fit these into the relevant points
mu(Ind) = E;
%plot
figure,
imagesc([11.15 24.15], [69 55.4], mu, 'alphadata', Ind)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off; colorbar
title('Predictions (OLS)')
