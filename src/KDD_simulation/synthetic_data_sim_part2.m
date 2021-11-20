clear all; close all; clc;
% Now we can run the data analytics on these stuffs...
load trial4_result_part1

X_reshape = zeros(M,K,Nt);
for n = 1 : Nt
    X_reshape(:,:,n) = X(n,:,:); 
end

lambda_est = zeros(Nt,1); 
for nn = 1 : Nt
    lambda_est(nn) = sum(vec(C(3:42,:,nn))) / K;
end

lambda_est = lambda_est * mean(lambda) / mean(lambda_est);
% It seems to be slightly off... --> by a constant scalar

P0_est = ones(L+3,M)/L;
X0_est = rand(M,K,Nt);
for nn = 1 : Nt
    for kk = 1 : K
        X0_est(:,kk,nn) = X0_est(:,kk,nn) / sum(X0_est(:,kk,nn));
    end
end
P_est = P0_est; 

X_est = X0_est;

% for nn = 1 : Nt
%     for kk = 1 : K
%         X_est(:,kk,nn) = X(nn,:,kk);
%     end   
% end

% C = Real_C;

% Let's try the ALS approach
for ao_iter = 1 : 20
    
    ao_iter
    % solve for "P"
    cvx_quiet(true);
    cvx_begin
        variable P_cvx(L+3,M)
        variable ls_term(Nt)
        for nn = 1 : Nt
            ls_term(nn) >= ...
                square_pos(norm(diag([ones(2,1); lambda_est(nn)*ones(L,1); 1])*P_cvx*X_est(:,:,nn) - C(:,:,nn),'fro'));
%                 + square_pos(norm(lambda_est(nn)*P_cvx(3:22,:)*X_est(:,:,nn) - C(3:22,:,nn),'fro'));
        end
        minimize( sum(ls_term) )
        subject to
            P_cvx >= 0; ones(1,L+3)*P_cvx == 1;
            P_cvx(1,1) == 0; P_cvx(1,3) == 0;
            P_cvx(2:22,3) == 0; % the priors on P
            P_cvx(23:42,2) == 0;
            P_cvx(3,1) == 0; P_cvx(42,1) == 0;
    cvx_end
    cvx_optval

    P_cvx = P;
    
    lambda_est = lambda;
    
    cvx_begin
        variable lambda_cvx(Nt)
        variable ls_term(Nt)
        for nn = 1 : Nt
            ls_term(nn) >= square_pos(norm(lambda_cvx(nn)*P_cvx(3:42,:)*X_est(:,:,nn) - C(3:42,:,nn),'fro'));
        end
        minimize( sum(ls_term) )
        subject to
            lambda_cvx >= 0;
    cvx_end
    lambda_est = lambda_cvx;
    
    % solve for "X"
    for nn = 1 : Nt
        cvx_begin
            variable X_cvx_nn(M,K)
            X_cvx_nn >= 0; ones(1,M)*(X_cvx_nn)==1;
            minimize( (norm(diag([ones(2,1); lambda_est(nn)*ones(L,1); 1])*P_cvx*X_cvx_nn - C(:,:,nn),'fro')))
            for kk = 1 : K
                if C(1,kk,nn) > 0
                    sum(X_cvx_nn(2:end,kk)) == 1;
                end
            end
        cvx_end
        X_est(:,:,nn) = X_cvx_nn;
    end
    
    sum(( vec(X_reshape - X_est) ).^2) / sum(vec(X).^2)
end

X_reshape = zeros(M,K,Nt);
for n = 1 : Nt
    X_reshape(:,:,n) = X(n,:,:); 
end

% Reshape X_est
X_data = zeros(Nt,(M-1)*K);
for nn = 1 : Nt
    X_data(nn,:) = vec( X_est(2:end,:,nn) );
end
