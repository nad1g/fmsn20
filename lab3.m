%
% lab-3
%

%kappa^2
k2 = 1e-2;

%local q structure
q1 = [0 -1 0;-1 4+k2 -1;0 -1 0];
q2 = [0 -0.1 -10; -0.1 20.4+k2 -0.1; -10 -0.1 0];
q3 = [0 -2 -4; -1 14+k2 -1; -4 -2 0];

m = 100; n = 100;
sz = [m,n];

%construct Q
Q1 = gmrfprec(sz,q1);
Q2 = gmrfprec(sz,q2);
Q3 = gmrfprec(sz,q3);

%covariances
%note: In a vector of size: size(Q1,1) x 1, 
%set the middle entry = (m*n/2 + m/2) to 1!
s1 = Q1\sparse(m*n/2 + m/2, 1, 1, size(Q1,1),1);
im = reshape(s1,m,n);
figure;
subplot(2,3,1), imagesc(im); colormap(gray);
s2 = Q2\sparse(m*n/2 + m/2, 1, 1, size(Q2,1),1);
im = reshape(s2,m,n);
subplot(2,3,2), imagesc(im); colormap(gray);
s3 = Q3\sparse(m*n/2 + m/2, 1, 1, size(Q3,1),1);
im = reshape(s3,m,n);
subplot(2,3,3), imagesc(im); colormap(gray);

%simulation
mu=5;
R = chol(Q1);
x = mu + R\randn(size(R,1),1);
subplot(2,3,4); imagesc(reshape(x,sz));colormap(gray);
R = chol(Q2);
x = mu + R\randn(size(R,1),1);
subplot(2,3,5); imagesc(reshape(x,sz));colormap(gray);
R = chol(Q3);
x = mu + R\randn(size(R,1),1);
subplot(2,3,6); imagesc(reshape(x,sz));colormap(gray);

%%
load('lab4.mat')
figure
subplot(1,2,1), imagesc(xtrue);
title('True Image')
subplot(1,2,2), imagesc(xmiss);
title('Image with missing parts')

Y=xmiss(known);
%A is the observation matrix whose entries are set to 1 if
%that pixel is known (and known is already given!)
Q = Q2;
A = sparse(1:length(Y),find(known), 1, length(Y), numel(xmiss));
Aall = [A ones(length(Y),1)]; %this is A_tilde
Qeps = 1e5*speye(length(Y));
Qbeta = 1e-6*speye(1);
Qall = blkdiag(Q, Qbeta);
%conditional expectation E(~x|y) [slide 11, lec 7]
Qxy = (Qall + Aall'*Qeps*Aall);

p = amd(Qxy);
Qxy = Qxy(p,p);
Aall = Aall(:,p);

Exy = Qxy\(Aall'*Qeps*Y);
Exy(p) = Exy;
Ezy = [speye(size(Q,1)) ones(size(Q,1),1)]*Exy;
sz = size(xtrue);
figure, imagesc(reshape(Ezy,sz))

%%
%sparse observation and interpolation
titan = imread('titan.jpg');
xtrue = double(titan);
miss=0.5;
known = rand(size(titan))>miss;
xmiss = xtrue.*known;
figure, subplot(121), imagesc(titan); colormap(gray)
subplot(122), imagesc(xmiss); colormap(gray)
Y=xmiss(known);

%local q structure, prec matrix
q = [0 -1 0;-1 4+k2 -1;0 -1 0];
sz = size(xmiss);
Q = gmrfprec(sz,q);

%A is the observation matrix whose entries are set to 1 if
%that pixel is known (and known is already given!)
A = sparse(1:length(Y),find(known), 1, length(Y), numel(xmiss));
Aall = [A ones(length(Y),1)]; %this is A_tilde
Qeps = 1e5*speye(length(Y));
Qbeta = 1e-6*speye(1);
Qall = blkdiag(Q, Qbeta);
%conditional expectation E(~x|y) [slide 11, lec 7]
Qxy = (Qall + Aall'*Qeps*Aall);

p = amd(Qxy);
Qxy = Qxy(p,p);
Aall = Aall(:,p);

Exy = Qxy\(Aall'*Qeps*Y);
Exy(p) = Exy;
Ezy = [speye(size(Q,1)) ones(size(Q,1),1)]*Exy;
sz = size(xtrue);
figure, imagesc(reshape(Ezy,sz))

