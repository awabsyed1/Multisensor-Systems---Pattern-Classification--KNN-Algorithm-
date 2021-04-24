%-------------------------------(Problem I (Part B)-----------------------%
%Extend the 1-nearest neighbours algorithm developed in Lab A.II to create
%k-nearest neighbours soltuions 
    % K = 1, 3 and 5 
    % Performed with only four FEATURES obtained during Lab A.I 

% Load Faults from Lab A.I (After Applying PCA to 4 Features)
load Fault_1 %Bearing Fault 
load Fault_2 % Gearmesh Fault 
load Fault_3 % Imbalance Fault 
load Fault_4 % Misalignment Fault 
load Fault_5 % Resonance Fault 

% load bearing_features.mat  % f_b 
% load gearmesh_features.mat  % f_g 
% load imbalance_features.mat %f_i
% load misalignment_features.mat %f_m
% load resonance_features.mat  %f_r

%Training Data - used in construction of the classifier 
%Test Data - used to evaluate the goodness or performance of the classifier
NoOfTrainingCases = 35;
NoOfTestingCases = length(Fault_1) - NoOfTrainingCases; % 15 vectors
% Trainging Data
trainingSet = [Fault_1(1:NoOfTrainingCases,:);
    Fault_2(1:NoOfTrainingCases,:); 
    Fault_3(1:NoOfTrainingCases,:);
    Fault_4(1:NoOfTrainingCases,:); 
    Fault_5(1:NoOfTrainingCases,:)]; %Training Data
% Testing set by last 15 vectors
testingSet = [Fault_1(NoOfTrainingCases+1:end,:);
    Fault_2(NoOfTrainingCases+1:end,:);
    Fault_3(NoOfTrainingCases+1:end,:); 
    Fault_4(NoOfTrainingCases+1:end,:); 
    Fault_5(NoOfTrainingCases+1:end,:)];  %Testing Data 
%----------Initial Varaiables for k-nearest neighbour search------------%
% Label sets 
trainingTarget = [ones(1,NoOfTrainingCases),... 
    ones(1,NoOfTrainingCases)*2,...
    ones(1,NoOfTrainingCases)*3,...
    ones(1,NoOfTrainingCases)*4,...
    ones(1,NoOfTrainingCases)*5];

testingTarget = [ones(1,NoOfTestingCases),... 
    ones(1,NoOfTestingCases)*2,...
    ones(1,NoOfTestingCases)*3,...
    ones(1,NoOfTestingCases)*4,...
    ones(1,NoOfTestingCases)*5];
%----------------------------k-nearest neighbour search-----------------%
% Total number of cases
totalNoOfTestingCases = NoOfTestingCases * 5;
totalNoOfTrainingCases = NoOfTrainingCases * 5;

inferredlabels = zeros(1,totalNoOfTestingCases); 

% This loop cycles through each unlabelled item:
for unlabelledCaseIdx = 1:totalNoOfTestingCases
    unlabelledCase = testingSet(unlabelledCaseIdx,:);
    
    % As any distance is shorter than infinity
    shortestDistance = inf;
    % Assign a temporary label
    shortestDistanceLabel = 0; 
    
    currentDist = 0;

    % This loop cycles through each labelled item:
    for labelledCaseIdx = 1:totalNoOfTrainingCases
        labelledCase = trainingSet(labelledCaseIdx, :);
        % Calculate the Euclidean distance:
        currentDist(labelledCaseIdx) = euc(unlabelledCase, labelledCase);
    end
    % find the 3 shortest distances
    [shortestDistance,I] = mink(currentDist,5); % 3 for k =3 
    % match the 3 shortest distances with the corresponding labels
    shortestDistanceLabel_temp = trainingTarget(I);
    
    % Most frequent label of k (k = 3 in this case) shortest distances
    % Mode used for most frequenct values 
    [shortestDistanceLabel,F] = mode(shortestDistanceLabel_temp);
    
    % if all 3 shortest distances fall into different fault types,
    % label the testing data as the type with shortest distance
    if F == 1
        shortestDistanceLabel = shortestDistanceLabel_temp(1);
    end
    
% Assign the found label to the vector of inferred labels:
inferredLabels(unlabelledCaseIdx) = shortestDistanceLabel;
end

% No. of correctly classified samples 
Nc = length(find(inferredLabels == testingTarget));
% No. of all samples 
Na = length(testingTarget);

%Accuracy of Classification ACC 
Acc = 100*(Nc/Na); % Accuracy of classification (ACC) 
disp(Acc)


%%
%-------------------------------Problem II (Assignment)-------------------%
% ---------PART A-----------%

%Description ; Wind Turbine manufacturing company & tasked to design a
%   multisensor signal estimation & health monitoring system for the blade
%   pitching mechanism of the wind turbine.

%Observation model y(t) = x(t) + v(t) (Part A)
%           x(t) = ax(t-1) + w(t)

%Prior Knowledge 
% sensor noise v ~ N(0,9)
% X - Uniformly distributed (0,30) 

load encoder.mat 

T = 100; % Data (Encoder) rows / measurement
var_noise = 9; %variance 
max = 30; 
min = 0;

k = 1;

for T = [1: 100]  % Case 1 = [1:100]  & Case 2 = [1:5] 
    t = var_noise/T; 
    t1 = sqrt(2*pi*t);
    x_mean = mean(encoder(1:T)); 
    
    numerator = @(x) ((x/t1) .* exp(-((x-x_mean).^2 / (2*t))));
    denominator = @(x) ((1/t1) .* exp(-(x-x_mean).^2 / (2*t)));
    
    num_int = integral(numerator,0,30); % 0 & 30 - limits 
    den_int = integral(denominator,0,30); % 0 & 30 - limits 
    
    x_MMSE(k) = num_int / den_int;
    k = k+1;
    plot (x_MMSE); hold on 
    xlabel ('Measurement') 
    ylabel ('MMSE')
    title ('MMSE wrt Measurement - Case 2 = [1:5] ')
    plot(x_mean,'o'); 
    legend('MMSE','Mean');hold off
end 
%%
%----------PART B--------------------%
% Prior knowledge X is Gaussian distributed N(15,4)
% Bayesian Estimation with gaussian Priors 
load encoder.mat 
var_x=4;
mean_x=15;
i=1;

x_MMSE_Gaus = zeros(1,2); %Pre-allocation for speed

for T=[1: 100]  % Case 1 = [1:100]  & Case 2 = [1:5] 
    x_mean=mean(encoder(1:T));
    x_MMSE_Gaus(i)=(x_mean/var_noise + mean_x/var_x)/(T/var_noise + 1/var_x);
    i=i+1;
    plot (x_MMSE_Gaus);  hold on 
    xlabel ('MMSE') 
    ylabel ('Measurement')
    title ('MMSE wrt Measurement - Case 2 - [1:5] ')
   plot(x_mean,'--'); hold off
end
%%
%--------------------PART C-----------------%
load straingauge.mat;

theta0 = 3000;
sigma0 = 1;
threshold=20;
straingauge = [0,straingauge];
T = length(straingauge);

g(1)=0; % Initialization / Starts from zero value

%NOTE: Possible drift(gamma) is ignored 
gamma = 0; % Therefore gamma = 0 
g(1) = 0;

for j=1:T-1
    g(j+1)=g(j)+(straingauge(j+1)-theta0)/sigma0 + gamma; 
    if abs(g(j+1)) > threshold % Crosses the threshold
        fault_on_set_time = j+1; %Fault occurs / Note the fault on set time
        disp(['Fault occured at ',num2str(fault_on_set_time)])  
    end
end
%%
%% Problem 2 (c)
% two-sided CUSUM test
% load data
load straingauge.mat
straingauge = [0,straingauge];
% normal operation condition
theta0 = 3000;
sigma0 = 1;
% threshold
delta = 20;
delta_neg = -20;
% data length
T = length(straingauge);
% no model error/drift
gamma = 0;

% 2 one-sided CUSUM test
% decision statistic
g(1) = 0;
G(1) = 0;
%%
% Positive CUSUM
disp('One-Sided Positive CUSUM Test:');
for i = 2:T
    g(i) = g(i-1) + (straingauge(i)-theta0)/sigma0 - gamma;
    % check if positive
    if g(i) < 0
        g(i) = 0;
    end
    % decision rule
    if g(i) > delta
        fprintf('Alert! Fault occurs at index: %i (T = %i).\n\n',i,i-1);
        break
    end
end
%%
% Negative CUSUM
disp('One-Sided Negative CUSUM Test:');
for i = 2:T
    G(i) = G(i-1) + (straingauge(i)-theta0)/sigma0 - gamma;
    % check if negative
    if G(i) > 0
        G(i) = 0;
    end
    % decision rule
    if G(i) < delta_neg
        fprintf('Alert! Fault occurs at index: %i (T = %i).\n\n',i,i-1);
        break
    end
end

%% Two sided test
disp('Two-Sided CUSUM Test:');
for i = 2:T
    g(i) = g(i-1) + (straingauge(i)-theta0)/sigma0 - gamma;
    % check if positive
    %if g(i) < 0
        %g(i) = 0;
    %end
    % decision rule
    if abs(g(i)) > delta
        fprintf('Alert! Fault occurs at index: %i (T = %i).\n\n',i,i-1);
        %break
    end
end