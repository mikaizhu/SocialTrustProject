%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       BTT Social Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all% Dont forget to initialize CVX

addpath('./jsonlab/')
addpath('./Tools/')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract Facebook data from JSON file 

fname=sprintf('TechAv.json');
if(exist(fname,'file')==0) break; end
fprintf(1,'===============================================\n>> %s\n',fname);
data=loadjson(fname);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find the most liked and commented posts
% Precise the data and the minimum of (likes+comments) per post
    
best_posts=get_best_posts(data,10); % the 2nd argument is the threshold for becoming a best_posts

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build a table of agents
%fields: ID, name, activity (Number of likes and comments he did)

agents =get_agents(best_posts);
agents = sort_agents (agents);

for i=1:length(agents)
    agents{i,2}= strcat('A' , int2str(i));
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Build the dictionary of the keywors we want to parse
% % Modify as it pleases you with your own keywords ang groups
% clear all; close all; clc;
% load easy_process_data
[dico,nb_groups] = dictionary();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build the term-document matrix here!
C = term_doc(dico,best_posts,agents);

L = length(dico)+1; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% One can immediately infer lambda from "C"
lambda = zeros(size(agents,1),1); lam_text = lambda;
for nn = 1 : length(lambda)
    lambda(nn) = sum(vec(C(3:L-1,:,nn))) / length(best_posts);
    lam_text(nn) = sum(vec(C(L,:,nn))) / length(best_posts);
end

M = nb_groups+1; K = length(best_posts);
X0 = parse(dico,nb_groups,best_posts,agents);
X0 = normalize(X0,K,M-1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filter out the agents who never speak up!
user_not_silent = find(lambda+lam_text > 0);
C = C(:,:,user_not_silent); 
lambda = lambda(user_not_silent); 
lam_text = lam_text(user_not_silent);
agents = agents(user_not_silent,:);
X0 = X0(user_not_silent,:);

N = length(lambda);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Can now do the inference based on the Poisson model --> For simplicity,
% we use the Least Square formulation

P0 = ones(L,M)/L;
% X0 = rand(M,K,N);
% for nn = 1 : N
%     for kk = 1 : K
%         X0(:,kk,nn) = X0(:,kk,nn) / sum(X0(:,kk,nn));
%     end
% end
X0_tmp = zeros(M,K,N);
for nn = 1 : N
    for kk = 1 : K
        X0_tmp(2:3,kk,nn) = X0(nn,(kk-1)*nb_groups+1:kk*nb_groups);
        X0_tmp(1,kk,nn) = 1 - sum(X0_tmp(2:3,kk,nn));
    end
end
X0 = X0_tmp;

P = P0; X = X0;

% Let's try the ALS approach
for ao_iter = 1 : 5
    ao_iter
    % solve for "P"
    cvx_quiet(true);
    cvx_begin
        variable P_cvx(L,M)
        variable ls_term(N)
        for nn = 1 : N
            ls_term(nn) >= square_pos(norm(P_cvx([1:2],:)*X(:,:,nn) - C([1:2],:,nn),'fro'))...
                + square_pos(norm(diag([lambda(nn)*ones(L-3,1); lam_text(nn)])*P_cvx(3:L,:)*X(:,:,nn) - C(3:L,:,nn),'fro'));
        end
        minimize( sum(ls_term) )
        subject to
            P_cvx >= 0; ones(1,L)*P_cvx == 1;
            P_cvx(1,1) == 0; 
            P_cvx(1,3) == 0;
            P_cvx(2:10,3) == 0; % the priors on P
            P_cvx(11:end-4,2) == 0;
            % P_cvx*ones(3,1) >= 0.01;
    cvx_end
    cvx_optval
    
    % solve for "X"
    for nn = 1 : N
        cvx_begin
            variable X_cvx_nn(M,K)
            X_cvx_nn >= 0; ones(1,M)*(X_cvx_nn)==1;
            minimize( square_pos(norm(P_cvx([1:2],:)*X_cvx_nn - C([1:2],:,nn),'fro'))...
                + square_pos(norm(diag([lambda(nn)*ones(L-3,1); lam_text(nn)])*P_cvx(3:L,:)*X_cvx_nn - C(3:L,:,nn),'fro')) )
            for kk = 1 : K
                if C(1,kk,nn) == 1
                    sum(X_cvx_nn(2:end,kk)) == 1;
                end
            end
        cvx_end
        X(:,:,nn) = X_cvx_nn;
    end
end

% load processed_data3
% Reshape X
X_data = zeros(N,nb_groups*K);
for nn = 1 : N
    X_data(nn,:) = vec( X(2:end,:,nn) );
end

X_old = parse(dico,nb_groups,best_posts,agents);
X_old= normalize(X_old,K,nb_groups);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is some kind of graph of the social network
% G(i,k) is the number of posts where i and k interacted.
[G,H]=get_graph(best_posts,agents);

% Now please save the data and run the program for Step2.m