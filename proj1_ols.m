%
% Ordinary Least Squares
%
% load data
load HA1_SE_Temp
sz = [273 260];
long = reshape(SweGrid(:,1), sz);
lat = reshape(SweGrid(:,2), sz);
elev = reshape(SweGrid(:,3), sz);
dist = reshape(SweGrid(:,4), sz);
dist_swe = reshape(SweGrid(:,5), sz);
%points inside of Sweden (i.e. not nan)
Ind = ~isnan(long);

plot_covariates = 0;
select_model = 0;
%%
% plot covariates
if plot_covariates == 1
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
end
%%
if select_model == 1
    % select model
    y = SweObs(:,6);
    % model 1: intercept + latitude + elevation + dist coast + dist swe coast
    disp('y ~ intercept + latitude + elevation + dist coast + dist swe coast')
    A = [ones(250,1) SweObs(:,[2,3,4,5])];
    [beta_,resid,sigma2,Sigma]= ols(A,y);
    disp(['SE of residuals = ',num2str(sqrt(sigma2))]);
    % model 2: intercept + latitude + elevation + dist swe coast
    disp('y ~ intercept + latitude + elevation + dist swe coast')
    A = [ones(250,1) SweObs(:,[2,3,5])];
    [beta_,resid,sigma2,Sigma]= ols(A,y);
    disp(['SE of residuals = ',num2str(sqrt(sigma2))]);
    % model 3: intercept + latitude + elevation
    disp('y ~ intercept + latitude + elevation')
    A = [ones(250,1) SweObs(:,[2,3])];
    [beta_,resid,sigma2,Sigma]= ols(A,y);
    disp(['SE of residuals = ',num2str(sqrt(sigma2))]);
    % model 4: 
    disp('y ~ intercept + latitude + dist swe coast')
    A = [ones(250,1) SweObs(:,[2,5])];
    [beta_,resid,sigma2,Sigma]= ols(A,y);
    disp(['SE of residuals = ',num2str(sqrt(sigma2))]);
end

%%
% estimate model parameters
% selected model is A: 
% y ~ intercept + latitude + elevation + dist coast + dist swe coast
I_val = datasample(1:250,25,'Replace',false);
I_obs = 1:250; I_obs(I_val) = [];

y = SweObs(I_obs,6);
A = [ones(length(I_obs),1) SweObs(I_obs,[2,3,4,5])];
[beta_,resid,sigma2,Sigma]= ols(A,y);
fprintf('-- Ordinary Least Squares --\n');
fprintf('Number of observations = %d\n', length(I_obs));
fprintf('Variance of the residuals = %f\n', sigma2);
seBeta = sqrt(diag(Sigma));
for ii = 1:length(seBeta),
    fprintf('beta_%d = %f, SE (beta_%d) = %f\n', ii, beta_(ii), ii, seBeta(ii));
end

% validation
I_val = sort(I_val);
Y_true = SweObs(I_val,6);
A_val = [ones(length(I_val),1) SweObs(I_val, [2,3,4,5])];
Y_v = A_val * beta_;
Vbeta_ = diag(Sigma);
Vmu = sum((A_val*Sigma).*A_val,2);
figure, plot(Y_true,'k--*');
hold on;
plot(Y_v,'r-o');
ci_low_mu = Y_v + 1.96*sqrt(Vmu);
ci_hi_mu = Y_v - 1.96*sqrt(Vmu);
%plot(ci_low_mu,'b-.');
%plot(ci_hi_mu,'b-.');
ci_low_Y_v = Y_v + 1.96*sqrt(Vmu + sigma2);
ci_hi_Y_v = Y_v - 1.96*sqrt(Vmu + sigma2);
plot(ci_low_Y_v,'m--.');
plot(ci_hi_Y_v,'m--.');
legend('Truth','Prediction','95% CI (low)', '95% CI (high)')
xlabel('Validation Locations')
ylabel('Temperature')
title('Prediction intervals for OLS')

%%
% do 'prediction'
% pick out the relevant parts of the grid
grid = SweGrid(Ind,:);
% create a matrix holding "predicitons"
mu = nan(sz);
A_grid = [ones(length(Ind(:)),1) grid(:,[2,3,4,5])];
E = A_grid*beta_;
%and fit these into the relevant points
mu(Ind) = E;
%plot
figure,
imagesc([11.15 24.15], [69 55.4], mu, 'alphadata', Ind)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off; colorbar
title('Predictions (OLS)')

% calculate variances
Vmu = sum((A_grid*Sigma).*A_grid,2);
se_pred = sqrt(Vmu + sigma2);
se = nan(sz);
se(Ind) = se_pred;
figure,
imagesc([11.15 24.15], [69 55.4], se, 'alphadata', Ind)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off