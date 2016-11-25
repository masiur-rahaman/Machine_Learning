function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % 'MarkerSize'

    temp=theta'*X';
    %disp(size(temp-y'));
    %disp(size(X(:,1))');
    %disp(m);
    theta_tmp_1=theta(1) - (alpha*sum((temp-y').*(X(:,1))'))/m;
    theta_tmp_2=theta(2) - (alpha*sum((temp-y').*(X(:,2))'))/m;
    theta(1) = theta_tmp_1;
    theta(2) = theta_tmp_2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
    %iteration_vect=[1:1:num_iters];
    %plot(iteration_vect, J_history);
    
    %fprintf('Program paused. Press enter to continue.\n');
    %pause;

end
