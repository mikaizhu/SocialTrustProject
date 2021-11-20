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
    
best_posts=get_best_posts(data,15); % the 2nd argument is the threshold for becoming a best_posts

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% One can immediately infer lambda from "C"
lambda = zeros(size(agents,1),1); 
for nn = 1 : length(lambda)
    lambda(nn) = sum(vec(C(:,:,nn))) / length(best_posts);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filter out the agents who never speak up!
user_not_silent = find(lambda > 0);
C = C(:,:,user_not_silent); lambda = lambda(user_not_silent); 
agents = agents(user_not_silent,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Can now do the inference based on the Poisson model --> For simplicity,
% we use the Least Square formulation
L = length(dico)+1; M = nb_groups+1; K = length(best_posts);
N = length(lambda);

P0 = ones(L,M)/L;
X0 = rand(M,K,N);
for nn = 1 : N
    for kk = 1 : K
        X0(:,kk,nn) = X0(:,kk,nn) / sum(X0(:,kk,nn));
    end
end

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
            ls_term(nn) >= square_pos(norm(P_cvx([1:2, L],:)*X(:,:,nn) - C([1:2, L],:,nn),'fro'))...
                + square_pos(norm(lambda(nn)*P_cvx(3:L-1,:)*X(:,:,nn) - C(3:L-1,:,nn),'fro'));
        end
        minimize( sum(ls_term) )
        subject to
            P_cvx >= 0; ones(1,L)*P_cvx == 1;
            P_cvx(1,1) == 0; 
%             P_cvx(1,3) == 0;
            P_cvx(2:12,3) == 0; % the priors on P
            P_cvx(13:end-2,2) == 0;
    cvx_end
    cvx_optval
    
    % solve for "X"
    for nn = 1 : N
        cvx_begin
            variable X_cvx_nn(M,K)
            X_cvx_nn >= 0; ones(1,M)*(X_cvx_nn)==1;
            minimize( square_pos(norm(P_cvx([1:2, L],:)*X_cvx_nn - C([1:2, L],:,nn),'fro'))...
                + square_pos(norm(lambda(nn)*P_cvx(3:L-1,:)*X_cvx_nn - C(3:L-1,:,nn),'fro')) )
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build the mask of B from the graph matrix G
N=length(agents(:,1));
N_s=30; % You can change this, don't worry
N_i=N-N_s;

B_MASK = G(N_s+1:N_i+N_s,1:N_s)>0;
D_MASK = G(N_s+1:N_i+N_s,N_s+1:N_s+N_i)>0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get B and D (Finally... wooof)

%using CVX 
[B,D]= do_the_magic_cvx(B_MASK,D_MASK,X_data,N_i,N_s,0.3); 

%using CVX 
[B_old,D_old]= do_the_magic_cvx(B_MASK,D_MASK,X_old,N_i,N_s,0.3); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Normalize

%D= normalize(D,1,N);
%B= normalize(B,1,N_s);

C=[eye(N_s,N_s),zeros(N_s,N_i);B,D];
figure; pcolor(C)
S = strcat('Number of stubborn = ' , int2str(N_s));
title(S)

figure;
C=[eye(N_s,N_s),zeros(N_s,N_i);B_old,D_old];
pcolor(C)
S = strcat('Number of stubborn = ' , int2str(N_s));
title(S)
%%%%%%%%%%%%%%%%% Have fun, but not too much!! %%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%% Trashbin %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% alpha = 0.0001;
% for grad_iter = 1 : 100
%     % gradient eval for "P"
%     gP = zeros(L,M);
%     for ll = 1 : L
%         for nn = 1 : N
%             for kk = 1 : K
%                 if ll <= 2
%                     gP(ll,:) = gP(ll,:) + (1-C(ll,kk,nn)/(P(ll,:)*X(:,kk,nn)))*X(:,kk,nn)';
%                 else
%                     gP(ll,:) = gP(ll,:) + (lambda(nn)-C(ll,kk,nn)/(P(ll,:)*X(:,kk,nn)))*X(:,kk,nn)';
%                 end
%             end
%         end
%     end
%     % grad eval for "X"
%     gX = zeros(M,K,N);
%     for nn = 1 : N
%         for kk = 1 : K
%             for ll = 1 : L
%                 if ll <= 2
%                     gX(:,kk,nn) = gX(:,kk,nn) + (1-C(ll,kk,nn)/(P(ll,:)*X(:,kk,nn)))*P(ll,:)';
%                 else
%                     gX(:,kk,nn) = gX(:,kk,nn) + (lambda(nn)-C(ll,kk,nn)/(P(ll,:)*X(:,kk,nn)))*P(ll,:)';
%                 end
%             end
%         end
%     end
%     
%     % projected gradient
%     X = min(max(0,X - alpha*gX),1);
%     P = min(max(0,P - alpha*gP),1);
% end
 