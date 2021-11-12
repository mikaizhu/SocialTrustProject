function [hatB,hatD] = solve_tmp(X,n_stubb,n_norm,rho,theta,gamma,L)

X = sparse(double(X));
[rows_num,~] = size(X);
Z = X(1:n_stubb,:);
Y = X(n_stubb+1:rows_num,:);

cvx_begin
cvx_solver sedumi
variable hatB(n_norm,n_stubb) nonnegative
variable hatD(n_norm,n_norm) nonnegative
variable temp
minimize(temp)
subject to
[hatB, hatD]*ones(n_norm+n_stubb, 1) == ones(n_norm, 1);
for i = 1:n_norm
	hatD(i, i)== L;
end
temp >= rho*norm((eye(n_norm)-hatD)*Y-hatB*Z,'fro')+theta*norm([hatB,hatD],'fro');
%temp >= rho*norm((eye(n_norm)-hatD)*Y-hatB*Z,'fro')+theta*norm(hatB,'fro')+gamma*norm(hatD,'fro');

cvx_end
