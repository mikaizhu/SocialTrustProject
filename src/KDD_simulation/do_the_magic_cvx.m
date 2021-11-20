function [B,D,error_fro]= do_the_magic_cvx(B_MASK,D_MASK,X,N_i,N_s,lambda)
          
%%%%%%% part 2 of the simulation --> regression on the data %%%%%%
% we want to infer B,D from the data
B_mask = B_MASK;
BC_mask = logical(1-B_MASK);
D_mask = D_MASK;
DC_mask = logical(1-D_MASK);

% Compute Y*pinv(Z)
YZ = X(N_s+1:end,:)*((X(1:N_s,:)*X(1:N_s,:)')^-1*X(1:N_s,:))';

%%%%%%%%%%%%%%%%% We use CVX here... %%%%%%%%
% B = B_normalize;
cvx_quiet(true)
cvx_solver('sedumi')
cvx_begin
    variable D(N_i,N_i) 
    variable B(N_i,N_s)
    minimize ( lambda*(norm(vec(D),1)+norm(vec(B),1)) + square_pos(norm(YZ - D*YZ - B, 'fro')) );
    subject to
        % op_exp_result(N_s+1:end,:) corresponds to the final opinions of
        % Non-stubborn agents, i.e., the matrix $Y$
        % op_exp_result(1:N_s,:) corresponds to the final/initial opinions
        % of stubborn agents, i.e., the matrix $Z$
%         square_pos(norm(YZ - D*YZ - B, 'fro')) <= 0.1;
        D(:) >= 0; B(:) >= 0;
        diag(D) == 0;
        B(BC_mask) == 0;
        D(DC_mask) == 0;
        ones(N_i,1) == [B D]*ones(N_i+N_s,1);
        
        
cvx_end

%%%%%%%%%%% the algorithm ends here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y = X(N_s+1:end,:); Z = X(1:N_s,:);

if (sum(isnan(vec(D)) + sum(isnan(vec(B)))))
    error_fro = inf;
else
    error_fro = norm(Y - pinv(eye(N_i)-D)*B*Z, 'fro')^2 / norm(Y,'fro')^2;
end
% error_fro = norm(YZ - D*YZ - B, 'fro')^2 / norm(YZ,'fro')^2;

end