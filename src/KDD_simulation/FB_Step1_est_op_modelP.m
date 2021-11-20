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
cnt_per_user = zeros(size(agents,1),1); 
for nn = 1 : length(cnt_per_user)
    cnt_per_user(nn) = sum(vec(C(:,:,nn)));
end

M = nb_groups; K = length(best_posts);
dico_old = dictionary_old_P();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filter out the agents who never speak up!
user_not_silent = find(cnt_per_user > 0);
C = C(:,:,user_not_silent); 
agents = agents(user_not_silent,:);
N = length(user_not_silent);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Can now do the inference based on the Poisson model --> For simplicity,
% we use the Least Square formulation

% These are the initialization part
P0 = zeros(L,M);
% Building P0 from  dicoP
dicoP = dictionary_old_P();
for l = 3 : L-1 % process everything except for the "like" & post action
    % find out what's the keyword
    for m = 1 : length(dicoP)
        if strcmp(dicoP(m).keyword,dico(l).keyword)
            P0(l,dicoP(m).group) = dicoP(m).value;
        end
    end
end
P0(1,1) = 0.1; % we hv to assign some prob for initiating a post..
P0(2,dicoP(1).group) = dicoP(1).value; % This corresponds to "like"
for m = 1 : length(dicoP) % this correspnod to *TEXT ONLY*
    if strcmp(dicoP(m).keyword,dico(2).keyword)
        P0(L,dicoP(m).group) = dicoP(m).value;
    end
end
% normalize P0
P0 = P0 ./ repmat(ones(1,L)*P0,L,1);

X0 = zeros(M,K,N);
total_cnt_words = sum(C(3:end,:,:),3)*ones(K,1);
lambda0 = total_cnt_words / max(total_cnt_words);

lambda_agent0 = zeros(N,1); 
for nn = 1 : N
    lambda_agent0(nn) = sum( vec( C(3:end,:,nn) ) ) / length(best_posts);
end

P = P0; X = X0; lambda = lambda0;
lambda_agent = lambda_agent0;

cvx_quiet(true);
% Let's try the ALS approach
for ao_iter = 1 : 5
    ao_iter
    
    % solve for "X"
    for nn = 1 : N
        cvx_begin
            variable X_cvx_nn(M,K)
            X_cvx_nn >= 0; ones(1,M)*(X_cvx_nn)==1;
            minimize( square_pos(norm(P([1:2],:)*X_cvx_nn - C([1:2],:,nn),'fro'))...
                + square_pos(norm(lambda_agent(nn)*diag(lambda)*P(3:L,:)*X_cvx_nn - C(3:L,:,nn),'fro')) )
            for kk = 1 : K
                if C(1,kk,nn) == 1
                    X_cvx_nn(1,kk) == 1;
                end
            end
        cvx_end
        X(:,:,nn) = X_cvx_nn;
    end

    % solve for "P"
    cvx_begin
        variable P_cvx(L,M)
        variable ls_term(N)
        for nn = 1 : N
            ls_term(nn) >= square_pos(norm(P_cvx([1:2],:)*X(:,:,nn) - C([1:2],:,nn),'fro'))...
                + square_pos(norm(lambda_agent(nn)*P_cvx(3:L,:)*X(:,:,nn) - diag(1./lambda)*C(3:L,:,nn),'fro'));
        end
        minimize( sum(ls_term) )
        subject to
            P_cvx >= 0; ones(1,L)*P_cvx == 1;
%             P_cvx(1:2,1) == 0;  
            P_cvx(1:2,2) == 0;
            P_cvx(2:10,2) == 0; % the priors on P
            P_cvx(11:end-5,1) == 0;
    cvx_end
    P = P_cvx;
    cvx_optval
    
    % solve for lambda_agent(nn)
    cvx_begin
        variable lam_cvx(N)
        variable ls_term(N)
        for nn = 1 : N
            ls_term(nn) >= square_pos(norm(P([1:2],:)*X(:,:,nn) - C([1:2],:,nn),'fro'))...
                + square_pos(norm(lam_cvx(nn)*diag(lambda)*P(3:L,:)*X(:,:,nn) - C(3:L,:,nn),'fro'));
        end
        minimize( sum(ls_term) )
        subject to
            lam_cvx >= 0;
    cvx_end    
    lambda_agent = lam_cvx;

end

% load processed_data3
% Reshape X
X_data = zeros(N,nb_groups*K);
for nn = 1 : N
    X_data(nn,:) = vec( X(1:end,:,nn) );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is some kind of graph of the social network
% G(i,k) is the number of posts where i and k interacted.
[G,H]=get_graph(best_posts,agents);

% Now please save the data and run the program for Step2.m