% clear all; close all; clc;
%%%%% Suppose that we have estimated "X"
load processed_data4
addpath('./Tools/')

% Reshape X
X_data = zeros(N,nb_groups*K);
for nn = 1 : N
    X_data(nn,:) = vec( X(2:end,:,nn) ); % Only need to take the last two rows, as it lives on a simplex...
end

% The "naive" way of learning opinion
X_old = parse(dico,nb_groups,best_posts,agents);
X_old= normalize(X_old,K,nb_groups);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is some kind of graph of the social network
% G(i,k) is the number of posts where i and k interacted.
[G,H]=get_graph(best_posts,agents);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build the mask of B from the graph matrix G
N=length(agents(:,1));
N_s=45; % You can change this, don't worry
N_i=N-N_s;

B_MASK = G(N_s+1:N_i+N_s,1:N_s)>0;
D_MASK = G(N_s+1:N_i+N_s,N_s+1:N_s+N_i)>0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get B and D (Finally... wooof)

%using CVX 
[B,D,error_fro]= do_the_magic_cvx(B_MASK,D_MASK,X_data,N_i,N_s,0.5); 

%using CVX 
[B_old,D_old,error_fro_old]= do_the_magic_cvx(B_MASK,D_MASK,X_old,N_i,N_s,0.5); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Normalize

%D= normalize(D,1,N);
%B= normalize(B,1,N_s);

figure;
C=[eye(N_s,N_s),zeros(N_s,N_i);B,D];
% figure; 
subplot(1,2,1);
imagesc(C); axis xy
S = strcat('LSA Op: Number of stubborn = ' , int2str(N_s));
title(S)

% figure;
subplot(1,2,2);
C=[eye(N_s,N_s),zeros(N_s,N_i);B_old,D_old];
imagesc(C); axis xy
S = strcat('DICT Op: Number of stubborn = ' , int2str(N_s));
title(S)