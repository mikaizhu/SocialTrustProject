% Detect the stubborn agents by greedy algorithm
% min_{V_s} || \hat{Y} - (I-D)^-1 B \hat{Z} ||_F^2

clear all; close all; clc;
load processed_data4
addpath('./Tools/')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gather the data in the right format

% Reshape X
X_data = zeros(N,nb_groups*K);
for nn = 1 : N
    X_data(nn,:) = vec( X(2:3,:,nn) ); % Only need to take the last two rows, as it lives on a simplex...
end

% Find the graph, we will reorder them afterwards
% This is some kind of graph of the social network
% G(i,k) is the number of posts where i and k interacted.
[G,H]=get_graph(best_posts,agents);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Let the fun begin...
Vs = [];
% begin with an empty set of stubborn agents
obj_prev = inf;
for ns = 1 : N
    % now assume that we have ns stubborn agents
    cand_set = setdiff(1:N,Vs); obj_try = zeros(length(cand_set),1);
    parfor nnn = 1 : (N-ns+1)
        % enumerate through the remaining 
        Vs_try = union(Vs,cand_set(nnn));
        cVs_try = setdiff(1:N,Vs_try);
        
        X_try = X_data([Vs_try,cVs_try],:); 
        G_try = G([Vs_try,cVs_try],[Vs_try,cVs_try]);
        
        N_i=N-ns;
        B_MASK = G_try(ns+1:N_i+ns,1:ns)>0;
        D_MASK = G_try(ns+1:N_i+ns,ns+1:ns+N_i)>0;
        
        [B,D,error_fro]= do_the_magic_cvx(B_MASK,D_MASK,X_try,N_i,ns,0.5); 
        obj_try(nnn) = error_fro;
    end
    
    [obj_cur,idx] = min(obj_try);
    % add this index to Vs
    Vs = union(Vs, cand_set(idx));
    
    if obj_cur > obj_prev
        break; % we can't go further
    end
    obj_prev = obj_cur;
    fprintf('We are continuing... %i \n',ns);
end

cVs = setdiff(1:N,Vs);
ns = length(Vs);
X_fin = X_data([Vs,cVs],:); 
G_fin = G([Vs,cVs],[Vs,cVs]);    
N_i=N-ns;
B_MASK = G_fin(ns+1:N_i+ns,1:ns)>0;
D_MASK = G_fin(ns+1:N_i+ns,ns+1:ns+N_i)>0;
        
[B,D,error_fro]= do_the_magic_cvx(B_MASK,D_MASK,X_fin,N_i,ns,0.5); 

N_s = ns;
figure;
C=[eye(N_s,N_s),zeros(N_s,N_i);B,D];
imagesc(C); axis xy
S = strcat('LSA Op: Number of stubborn = ' , int2str(N_s));
title(S)
% 
% % figure;
% subplot(1,2,2);
% C=[eye(N_s,N_s),zeros(N_s,N_i);B_old,D_old];
% imagesc(C); axis xy
% S = strcat('DICT Op: Number of stubborn = ' , int2str(N_s));
% title(S)