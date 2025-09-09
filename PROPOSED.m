function [best_fitness, convergence_curve, best_solution, time] = PROPOSED(X, func, lb, ub, max_iter)
% Initialize parameters

% Initialize population matrix X
[N, dim] = size(X);

% Evaluate initial population fitness
F = (feval(func, X(:)));

% Initialize best fitness and position
[best_fitness, best_idx] = min(F);
best_solution =  X(dim, :);

convergence_curve = zeros(1, max_iter);
t=1;
% Main loop
for t = 1:max_iter
    for i = 1:N
        % Phase 1: Choosing and Moving to the Migration Destination (Exploration Phase)
        % Determine candidate destinations for migration
        candidate_destinations = X(F < F(i), :);
        if ~isempty(candidate_destinations)
            % Randomly choose a migration destination
            MD = candidate_destinations(randi(size(candidate_destinations, 1)), :);
            
            % Calculate new position of population member
            I = randi([1, 2], 1, dim); % Randomly select 1 or 2
            
            r = F(i) / ((mean(F) / max(F)) / (mean(F) / min(F))); %Traditional update
            
            X_P1 = X(i, :) + r .* (MD - I .* X(i, :));
            
            % Boundary control for new position
            X_P1 = max(min(X_P1, ub), lb);
            
            % Evaluate the new position
            F_P1 = feval(func, X_P1);
            
            % Update the ith population member if new position is better
            if F_P1 < F(i)
                X(i, :) = X_P1(i, :);
                F(i) = F_P1(i);
            end
        end
        
        % Phase 2: Adaptation to the New Environment (Exploitation Phase)
        r = r;
        X_P2 = X(i, :) + (1 - 2 * r) .* ((ub - lb) / t);
        
        % Boundary control for new position
        X_P2 = max(min(X_P2, ub), lb);
        
        % Evaluate the new position
        F_P2 = feval(func, X_P2);
        
        % Update the ith population member if new position is better
        if F_P2 < F(i)
            X(i, :) = X_P2(i, :);
            F(i) = F_P2(i);
        end
    end
    
    % Update the best solution found so far
    [current_best_fitness, best_idx] = min(F);
    if current_best_fitness < best_fitness
        best_fitness = current_best_fitness;
        best_solution = X(best_idx, :);
    end
    
    % Store convergence information
    convergence_curve(t) = best_fitness;
    
    % Display progress every 50 iterations
    if mod(t, 50) == 0
    end
end
t=t+1;
time=toc;
end