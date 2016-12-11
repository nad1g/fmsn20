function [Exy, Qxy] =  gmrf_reconstruct(Aall, Qall, Qeps, Y)
% [Exy, Qxy] = gmrf_reconstruct(At, Qt, Qeps, y) performs
% GMRF reconstruction.
% Inputs:
% Aall = [A B] : N x (N+K) matrix, A: observation Matrix B: matrix of covariates
% Qall = blockdiag(Q,Q_beta)
% Qeps = measurement error precision matrix
% Y = observations (Nx1)
% Exy = E_[x|y] posterior mean of the reconstructed latent field
% Qxy = prec matrix of posterior


% from slide 16 of lecture 7.
Qxy = (Qall + Aall'*Qeps*Aall);
p = amd(Qxy);
Qxy = Qxy(p,p);
Aall = Aall(:,p);
Exy = Qxy\(Aall'*Qeps*Y);
Exy(p) = Exy;

