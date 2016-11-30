%load data
load HA2_Parana.mat

%compute precision matrices for Matern-fields
[spde.C, spde.G, spde.G2] = matern_prec_matrices(mesh.loc, mesh.T);

figure(1)
subplot(221)
%triangulation
triplot(mesh.T, mesh.loc(:,1), mesh.loc(:,2), 'color', [.75 .75 .75])
hold on
%observation locations and borders (marking coast in different colour)
plot(mesh.loc_obs(:,1), mesh.loc_obs(:,2), '.', ...
    Border(:,1),Border(:,2),'-', ...
    Border(1034:1078,1),Border(1034:1078,2),'-')
hold off
axis tight

%observations
subplot(222)
scatter(mesh.loc_obs(:,1), mesh.loc_obs(:,2), 15, rainfall, ..., 
    'filled', 'markeredgecolor', 'k')
hold on
plot(Border(:,1),Border(:,2),'-',...
  Border(1034:1078,1),Border(1034:1078,2),'-')
colorbar; hold off; axis tight

%elevation
subplot(223)
trisurf(mesh.T, mesh.loc(:,1), mesh.loc(:,2),  ...
        zeros(size(mesh.loc,1),1), mesh.elevation);
hold on
scatter(mesh.loc_obs(:,1), mesh.loc_obs(:,2), 15, mesh.elevation_obs, ..., 
    'filled', 'markeredgecolor', 'k')
plot(Border(:,1),Border(:,2),'-',...
  Border(1034:1078,1),Border(1034:1078,2),'-')
view(0,90); shading interp; caxis([0 1500]); colorbar;
hold off; axis tight

%distance to coast
subplot(224)
trisurf(mesh.T, mesh.loc(:,1), mesh.loc(:,2),  ...
    zeros(size(mesh.loc,1),1), mesh.dist);
hold on
scatter(mesh.loc_obs(:,1), mesh.loc_obs(:,2), 15, mesh.dist_obs, ...
    'filled', 'markeredgecolor', 'k')
plot(Border(:,1),Border(:,2),'-',...
  Border(1034:1078,1),Border(1034:1078,2),'-')
view(0,90); shading interp; colorbar; 
hold off; axis tight

%plot data
figure(2)
subplot(231)
histfit(rainfall,[],'normal')
subplot(232)
histfit(log(rainfall),[],'normal')
subplot(233)
histfit(rainfall,[],'gamma')
%and qq-plots
subplot(234)
normplot(rainfall)
subplot(235)
normplot(log(rainfall))
subplot(236)
pars = gamfit(rainfall);
n = length(rainfall);
qqplot(rainfall, gaminv((1:n)/n,pars(1),pars(2)))

%illustrate sparsity (matrices already have near-optimal order)
figure(3)
subplot(231)
spy(spde.C)
subplot(232)
spy(spde.G)
subplot(233)
spy(spde.G2)
subplot(234)
spy(spde.C+2*spde.G+spde.G2)
subplot(235)
spy(chol(spde.C+2*spde.G+spde.G2))
