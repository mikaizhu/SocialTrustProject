% testing the algorithm...
clear all; close all; clc;

M = 3; N = 60; Ns = 35; % we have N+Ns agents in total
p = 0.1;
d_s = 5; % just random graphs.. can be changed later
K = 2*Ns; % just to ensure full rank-ness

L = 40; % no of actions in the dictionary (Plus `like', `post', `do nothing')
%% Gen. the opinion data
% It needs to respect Y = (I-D)^-1 BZ

% Ordinary agents
G = rand(N,N) <= p; % correspond to the normal user
G = triu(G,1); G = G + G'; G = G > 0;

% Stubborn agents
G_sn = zeros(Ns,N);
while (min(ones(1,Ns)*G_sn) == 0) % to ensure the assumption is satisfied
    G_sn = zeros(Ns,N);
    for nn = 1 : N
        G_sn(randperm(Ns,d_s),nn) = 1;
    end
end

% Gen the Trust matrix
G_com = [zeros(Ns) G_sn; G_sn' G];
W = (G_com+eye(N+Ns)) .* rand(N+Ns); % SE_G is the \overline{W} 

max_W = max(W) + 0.05;
W = W - diag(diag(W)) + diag(max_W);
W = W ./ repmat(W*ones(Ns+N,1),1,Ns+N);
W(1:Ns,1:Ns) = eye(Ns); W(1:Ns,Ns+1:end) = 0;
D = W(Ns+1:end,Ns+1:end); B = W(Ns+1:end,1:Ns);

% Gen the opinions
X0 = rand(N+Ns,M,K); X = zeros(N+Ns,M,K);
for k = 1 : K
    % normalize
    X0(:,:,k) = X0(:,:,k) ./ repmat(X0(:,:,k)*ones(M,1),1,M);
    % gen the final op
    X(1:Ns,:,k) = X0(1:Ns,:,k);
    X(Ns+1:end,:,k) = ((eye(N)-D)^-1) * B*X0(1:Ns,:,k);
end

% Gen the P matrix
P = rand(L+3,M); 
P(1,1) = 0; P(1,3) = 0; % the poster must be "yes"
P(2:22,3) = 0; % These actions cannot be `no', including the `like' & `post' action
P(23:42,2) = 0; % These actions cannot be `yes'
P(42,1) = 0; % the last action is a strong "no"
P(3,1) = 0; % the 1st comment is a strong "yes"
% P(1:end,1) = 0; % If I don't care, then I won't take any action w.p. 1

% P(2,:) = 10*P(2,:); % More motivation to pressing `like' 
P = P ./ repmat(ones(1,L+3)*P,L+3,1);

% Generate the term-document matrix
% Generate the term-document matrix
Nt = Ns + N;
C = zeros(L+3,K,Nt); % it includes the special action, except for the null action
Real_C = C;
lambda = 3000* rand(Nt,1); % assume lambda is fixed first and it's user specific

for s = 1 : K
    flag = 1; % I haven't found the poster
    while (flag)
        % find the poster
        for n = 1 : Nt
            flip_a_coin = ( rand <= P(1,:)*X(n,:,s)' );
            if flip_a_coin == 1
                % It is the poster
                C(1,s,n) = 1; Real_C(1,s,n) = P(1,:)*X(n,:,s)';
                X(n,1,s) = 0; X(n,:,s) = X(n,:,s) / sum( X(n,:,s) );
                flag = 0; break;
            end
        end
    end
    % Find someone to like
    for n = 1 : Nt
        C(2,s,n) = (rand <= P(2,:)*X(n,:,s)');
        Real_C(2,s,n) = P(2,:)*X(n,:,s)';
    end
    % Find someone to comment
    for n = 1 : Nt
        for l = 3 : 42
            C(l,s,n) = poissrnd( P(l,:)*( lambda(n)*X(n,:,s)' ) );
            Real_C(l,s,n) = P(l,:)*( lambda(n)*X(n,:,s)' );
        end
    end
    % Fill in the no-action field
    for n = 1 : Nt
        if sum(C(:,s,n)) == 0
            C(43,s,n) = 1;
        end
        
    end
end



