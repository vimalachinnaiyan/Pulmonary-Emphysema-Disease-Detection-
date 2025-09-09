function [best_fitness, cg_curve,  best_whale, time] = WOA(whales, fitness_func, Ub, Lb, max_iter)
% Walrus Optimization Algorithm (WOA)
ub = Ub(1,:);
lb = Lb(1,:);
[num_variables,num_agents] = size(whales);
cg_curve=zeros(1,max_iter);

% Initialize the positions of search agents
fitness = inf(num_agents, 1);
best_fitness = inf;
best_whale = nan(1, num_variables);
tic;
for iter = 1:max_iter
    for i = 1:num_agents
        % Calculate the fitness of each whale
        current_fitness = feval(fitness_func, whales(i,:));
        if current_fitness < fitness(i)
            fitness(i) = current_fitness;
            % Update the position of the best search agent
            if current_fitness < best_fitness
                best_fitness = current_fitness;
                best_whale = whales(i,:);
            end
        end
    end
    
    a = 2 - iter * (2 / max_iter); % a decreases linearly from 2 to 0
    
    for i = 1:num_agents
        A = 2 * a * rand() - a;
        C = 2 * rand();
        b = 1;               % Defines the shape of the spiral
        l = (rand() * 2 - 1) * 1; % l is a random number in [-1,1]
        p = rand();          % p in [0, 1]
        
        for j = 1:num_variables
            if p < 0.5
                if abs(A) >= 1
                    rand_leader_index = floor(num_agents * rand() + 1);
                    X_rand = whales(rand_leader_index, :);
                    D_X_rand = abs(C * X_rand - whales);
                    whales = X_rand - A * D_X_rand;
                else
                    D_Leader = abs(C * best_whale - whales);
                    whales = best_whale - A * D_Leader;
                end
            else
                distance_to_leader = abs(best_whale - whales);
                whales = distance_to_leader * exp(b * l) * cos(l * 2 * pi) + best_whale;
            end
        end
        
        % Ensure the whales stay within bounds
        whales(i,:) = max(whales(i,:), lb);
        whales(i,:) = min(whales(i,:), ub);
    end
end
time = toc;
best_fitness = cg_curve(end);
end
