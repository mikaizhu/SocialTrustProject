%facebok

clc;
clear;
load('matlab_result/solution1_X.mat');%40个stubb
load('matlab_result/solution2_X.mat');%40
load('matlab_result/solution3_X.mat');%40
X = solution3_X;
[nrow,ncols] = size(X);

n_stubb = 120;
n_norm = nrow-n_stubb;
tic
L = 0.1;%可根据讲话频率调整自信度
rho = 0.9;
theta = 0.2;
gamma = 0;
%X,n_stubb,n_norm,rho,theta,gamma,L
[B_hat,D_hat] = solve_tmp(X, n_stubb, n_norm, rho, theta, gamma, L);
%W_hat = [W(1:n_stubb,:);[B_hat,D_hat]];
figure(1)
I = eye(n_stubb);
O = zeros(n_stubb,n_norm);
W_hat = [I,O;B_hat,D_hat];
paint(W_hat,['W']);
save('matlab_result/W2.mat','W_hat');
figure(2)
paint(X,['X']);







