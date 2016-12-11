function pc = sample_pc(N,k,a,b)
A = k + a - 1;
B = N - k + b - 1;
pc = betarnd(A,B);

