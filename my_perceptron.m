% ---------------------------------------------------
%
%   MY_PERCEPTRON -
%
%
%   Created:          06.04.2016 (Alexander Kramlich)
%   Last modified:    08.04.2016 (Alexander Kramlich)
%
% ---------------------------------------------------

tic;

[M, N] = size(data);
N = N-1;

my_class = 9;
allowed_modes = {'Batch', 'Stochastic'};
mode = allowed_modes{1};

%lern_rate = [0.01:0.01:0.1, 0.2:0.1:1];
lern_rate = 0.01;
L = length(lern_rate);
my_factor = 1;

switch mode
    case 'Batch'
        N_epochs = 50;
    case 'Stochastic'
        N_epochs = 30;
    otherwise
end

%% Divide data into Training, Validataion and Test
N_training = 3000;
N_validation = 1000;
N_test = 1000;

training_data = data(1:N_training, :)';
training_data(2:end,:) = training_data(2:end,:)./my_factor;
training_data(2,:) = 1;

validation_data = data(N_training+1:N_training+N_validation, :)';
validation_data(2:end,:) = validation_data(2:end,:)./my_factor;
validataion_data(2,:) = 1;

test_data = data(N_training+N_validation+1:end, :)';
test_data(2:end,:) = test_data(2:end,:)./my_factor;
test_data(2,:) = 1;

w = ones(1,N);

training_performance = zeros(L,N_epochs);
training_true_positive = zeros(L,N_epochs);
training_false_negative = zeros(L,N_epochs);
validation_performance = zeros(L,N_epochs);
test_performance = zeros(L,N_epochs);

% Permutation of training data
p = randperm(N_training);

%% Adjust weights
switch mode
    case 'Batch'
        for l=1:L
            for n=1:N_epochs
                % Output of the current perceptron
                nn_o_training = activation_function(w*training_data(2:end,:));
                nn_o_validation = activation_function(w*validation_data(2:end,:));
                nn_o_test = activation_function(w*test_data(2:end,:));
                
                % Correct output
                nn_t_training = training_data(1,:)==my_class;
                nn_t_validation = validation_data(1,:)==my_class;
                nn_t_test = test_data(1,:)==my_class;

                % Adjustment
                w = w + lern_rate(l).*(nn_t_training - nn_o_training)*training_data(2:end,:)';

                % Training performance
                training_performance(l,n) = sum(nn_o_training ~= nn_t_training)/N_training;
                training_true_positive(l,n) = (sum((training_data(1,:)==my_class) & nn_o_training)/...
                                                sum(training_data(1,:)==my_class))*100;
                training_false_negative(l,n) = (sum((training_data(1,:)~=my_class) & nn_o_training)/...
                                                sum(training_data(1,:)~=my_class))*100;
                % Validation performance    
                validation_performance(l,n) = sum(nn_o_validation ~= nn_t_validation)/N_validation;
                % Test performance
                test_performance(l,n) = sum(nn_o_test ~= nn_t_test)/N_test;
            end
        end
    case 'Stochastic'
        validation_performance = zeros(1,N_epochs*N_training+1);
        validation_performance(1) = 1;
        for n=1:N_epochs
            for k=1:N_training
                nn_o_training = activation_function(w*training_data(2:end,p(k)));
                nn_o_validation = activation_function(w*validation_data(2:end,:));
                
                nn_t_training = training_data(1,p(k))==my_class;
                nn_t_validation = validation_data(1,:)==my_class;
                
                % Adjustment
                w_old = w;
                w = w + lern_rate*(nn_t_training - nn_o_training)*training_data(2:end,p(k))';
                
                % Validation performance    
                validation_performance((n-1)*N_training + k + 1) = sum(nn_o_validation ~= nn_t_validation)/N_validation;
                
                % If the performance of the NN worsens, keep the old weights
                if validation_performance((n-1)*N_training + k + 1) > validation_performance((n-1)*N_training + k)
                    w = w_old;
                end
            end
            fprintf('Epoch #%g\n', n)
        end
    otherwise
        error('No mode selected!')
end

fprintf('Time elapsed: %g ms\n', toc(tic)*1000);

%% Plot
close all
figure
hold on
switch mode
    case 'Batch'
        plot(training_performance(1,:))
    otherwise
end
plot(validation_performance(1,:))
title(sprintf('Performance of the Neural Network for Class %g', my_class))
xlabel('Epoch')
ylabel('Relative Error')
legend('Training Data', 'Validation Data')
hold off

switch mode
    case 'Batch'
        figure
        hold on
        plot(training_true_positive(1,:))
        plot(training_false_negative(1,:))
        title(sprintf('True Positives and False Positives for Class %g', my_class))
        xlabel('Epoch')
        ylabel('%')
        legend('True Positive', 'False Negative')
        hold off
    otherwise
end

if L>1
    figure
    surf(1:N_epochs, 1:L, test_performance)
    xlabel('Epoch')
    ylabel('Lern Rate')
end
zlabel('Relative Error')