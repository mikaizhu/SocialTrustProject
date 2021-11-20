clear all; close all; clc
% Part 3 of the simulation
% We essentially do detection of stubborn agents here
% load result_after_part2
load trial4_result_part2

% to find the graph from the term-doc matrix...
% from $C$???
% this is a sound assumption but does not really matter in the model!?
G_sim = ((G_com + (rand(Nt)<0.5)) > 0); 
% load result_part3_imm

% Reshape X_est
% X_data = zeros(N,(M-1)*K);
% for nn = 1 : Nt
% %     X_data(nn,:) = vec( X_est(2:end,:,nn) );
%     X_data(nn,:) = vec(X(nn,2:end,:));
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Let the fun begin...
Vs = [];
% begin with an empty set of stubborn agents
obj_prev = inf;
for ns = 1 : Nt
    % now assume that we have ns stubborn agents
    cand_set = setdiff(1:Nt,Vs); obj_try = zeros(length(cand_set),1);
    parfor nnn = 1 : (Nt-ns+1)
        % enumerate through the remaining 
        Vs_try = union(Vs,cand_set(nnn));
        cVs_try = setdiff(1:Nt,Vs_try);
        
        X_try = X_data([Vs_try,cVs_try],:); 
        G_try = G_sim([Vs_try,cVs_try],[Vs_try,cVs_try]);
        
        N_i=Nt-ns;
        B_MASK = G_try(ns+1:N_i+ns,1:ns)>0;
        D_MASK = G_try(ns+1:N_i+ns,ns+1:ns+N_i)>0;
        
        [B_est,D_est,error_fro]= do_the_magic_cvx(B_MASK,D_MASK,X_try,N_i,ns,10); 
        obj_try(nnn) = error_fro;
    end
    
    [obj_cur,idx] = min(obj_try);
    % add this index to Vs
    Vs = union(Vs, cand_set(idx));
    
%     if obj_cur > obj_prev
%         break; % we can't go further
%     end
    if obj_cur <= 0.01
        break;
    end
    obj_prev = obj_cur;
    fprintf('We are continuing... %i, obj: %f \n',ns,obj_cur);
    Vs
end

cVs = setdiff(1:Nt,Vs);
ns = length(Vs);
X_fin = X_data([Vs,cVs],:); 
G_fin = G_sim([Vs,cVs],[Vs,cVs]);    
N_i=Nt-ns;
B_MASK = G_fin(ns+1:N_i+ns,1:ns)>0;
D_MASK = G_fin(ns+1:N_i+ns,ns+1:ns+N_i)>0;
        
[B_est,D_est,error_fro]= do_the_magic_cvx(B_MASK,D_MASK,X_fin,N_i,ns,1); 

N_s = ns;
figure;
C=[eye(N_s,N_s),zeros(N_s,N_i);B_est,D_est];
% Reoder the matrix, such that it makes sense for comparison?
C_ord = C;
C_ord([Vs,cVs],[Vs,cVs]) = C;

imagesc(C_ord); axis xy
S = strcat('LSA Op: Number of stubborn = ' , int2str(N_s));
title(S)

W_reweight = W;
d = diag(W);
d = d(36:end);
W_reweight(36:end,36:end) = (eye(60)-diag(d))^-1 * W(36:end,36:end);
W_reweight(36:end,1:35) = (eye(60)-diag(d))^-1 * W(36:end,1:35);
W_reweight = W_reweight - diag(diag(W_reweight));
W_reweight(1:35,1:35) = eye(35);


figure; imagesc(W_reweight); axis xy
S = strcat('LSA Op: Number of stubborn = ' , int2str(N_s));
title(S)