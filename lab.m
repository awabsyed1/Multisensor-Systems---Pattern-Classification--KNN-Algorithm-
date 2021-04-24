% Author: Awabullah M Syed 
% Date: 12 March 2021 
% Description: Multisensor and Decision Systems of Test Rig: Bearing Defect
%   Gear Mesh, Resonance, Imbalance and Misalignment and extracting
%   informative features for health monitoring purpose 

% NOTE: Please run the code section-wise 

%-----------------------Lab A.I-------------------------------------------%

%-----------------------Task 1: Vibration Signal Analysis----------------%
%Load Files (ALWAYS RUN THIS SECTION)
load bearing.mat %Vibration MEAS of bearing-defect (Ts = 50sec,fs = 1 000)
load gearmesh.mat %Vibration MEAS of gearmesh rig (Ts = 50sec, fs = 1 000)
load misalignment.mat % MEAS of misalignment rig (Ts = 50sec, fs = 1 000)
load imbalance.mat % MEAS of imbalance rig (Ts = 50sec, fs = 1 000)
load resonance.mat % MEAS of resonance rig (Ts = 50sec, fs = 1 000)

Ts = 50; 
Fs = 1000; 
T = 1 / Fs; 
N = 50000; %Measurement 
t = (0:N-1); %time 
%%
%----------------------Time-Domain Plots----------------------------------%
featurename = {'bearing','gearmesh','misalignment','imbalance','resonance'};
feature = [bearing,gearmesh,misalignment,imbalance,resonance]; 
% Plot the figures of each fault in time domain
    for i=1:5
        subplot(5,1,i);
        plot(t,feature(:,i));
        xlabel('time'),ylabel(featurename{i});
        title(['Figure of ',featurename{i},' in time domain']);
    end

figure (1)
plot (t,bearing)  %plot for bearing measurement
title ('Time-Domain of bearing-defect rig')
movegui(figure(1),'southeast')
xlabel('Time sec') 
ylabel ('Sampled Measurement')
figure (2)
plot (t,gearmesh) % plot for gearmesh meassurements
title ('Time-Domain of gearmesh rig')
movegui(figure(2),'northeast')
xlabel('Time sec') 
ylabel ('Sampled Measurement')
figure (3)
plot (t,misalignment) % plot for misalignment rig meassurements
title ('Time-Domain of misalignment rig')
movegui(figure(3),'northwest')
xlabel('Time sec') 
ylabel ('Sampled Measurement')
figure (4)
plot (t,imbalance) % plot for imbalance rig meassurements
title ('Time-Domain of imbalance rig')
movegui(figure(4),'southwest')
xlabel('Time sec') 
ylabel ('Sampled Measurement')
figure (5)
plot (t,resonance) % plot for resonance rig meassurements
title ('Time-Domain of resonance rig')
xlabel('Time sec') 
ylabel ('Sampled Measurement')
movegui(figure(5),'north')
%--------------------Time-Domain subplot---------------------------------%
figure (6)  %Full-screen the plot for better viewing 
subplot(2,3,1)
plot (t,bearing)  %plot for bearing measurement
title ('Time-Domain of bearing-defect rig')
xlabel('Time sec') 
ylabel ('Sampled Measurement')
subplot(2,3,2)
plot (t,gearmesh) % plot for gearmesh meassurements
title ('Time-Domain of gearmesh rig')
xlabel('Time sec') 
ylabel ('Sampled Measurement')
subplot(2,3,3)
plot (t,misalignment) % plot for misalignment rig meassurements
title ('Time-Domain of misalignment rig')
xlabel('Time sec') 
ylabel ('Sampled Measurement')
subplot(2,3,4)
plot (t,imbalance) % plot for imbalance rig meassurements
title ('Time-Domain of imbalance rig')
xlabel('Time sec') 
ylabel ('Sampled Measurement')
subplot(2,3,5)
plot (t,resonance) % plot for resonance rig meassurements
title ('Time-Domain of resonance rig')
xlabel('Time sec') 
ylabel ('Sampled Measurement')
%%
%-----------------Frequency-Domain - Spectral Analysis------------------%
[P1,~] = pwelch(bearing,[],[],[],1000); %samping frequency of 1 kHz 
[P2,~] = pwelch(gearmesh,[],[],[],1000);
[P3,~] = pwelch(misalignment,[],[],[],1000);
[P4,~] = pwelch(imbalance,[],[],[],1000);
[P5,f] = pwelch(resonance,[],[],[],1000);
%Since all the frequency component of pwelch are same, therefore ~ is used
%       to speed the processing time 
P = [P1,P2,P3,P4,P5]; % PSD of all 5 
til = ["Bearing freq","Gearmesh freq","Misalignment freq","Imbalance freq","Resonance freq"];
% Frequency plot 
i = 7;
k = 1;
while i > 6 && i <12
figure (i)
plot(f,P(:,k))
xlabel ('Frequency (Hz)') 
ylabel ('Power Spectral Density Estimate') 
title ({til(1,k)})
i = i +1;
k = k +1;
end 
%%
%-----------------------------Lab A.I - Task II--------------------------%
% Task II : Feature Extraction
%--------------------Reshaping Matrices----------------------------------%
reshape_bearing = reshape(bearing,1000,50);
reshape_gearmesh = reshape(gearmesh,1000,50); 
reshape_imbalance = reshape(imbalance,1000,50);
reshape_misalignment = reshape(misalignment,1000,50);
reshape_resonance = reshape(resonance,1000,50); 
%--------------------------Pre-allocating for speed-----------------------%
x_normalb = zeros(1000,50); x_normalg = zeros(1000,50); 
x_normali = zeros(1000,50); x_normalm = zeros(1000,50);
x_normalr = zeros(1000,50);
 
%----------------------------Normalization--------------------------------%
for j = 1:50 
    xmean_b = repmat(mean(reshape_bearing(:,j)),1000,1);
    xmean_g = repmat(mean(reshape_gearmesh(:,j)),1000,1);
    xmean_i = repmat(mean(reshape_imbalance(:,j)),1000,1);
    xmean_m = repmat(mean(reshape_misalignment(:,j)),1000,1);
    xmean_r = repmat(mean(reshape_resonance(:,j)),1000,1);
    
    x_normalb(:,j) = reshape_bearing(:,j) - xmean_b; %Bearing 
    x_normalg(:,j) = reshape_gearmesh(:,j) - xmean_g; %GearMesh     
    x_normali(:,j) = reshape_imbalance(:,j) - xmean_i; % Imbalance 
    x_normalm(:,j) = reshape_misalignment(:,j) - xmean_m;% Misalignment 
    x_normalr(:,j) = reshape_resonance(:,j) - xmean_r; % Resonance 
    
end 
% saved so that it can be used to complete the assignment problem I (a)
save normalized_bearing x_normalb
save normalized_gearmesh x_normalg
save normalized_imbalance x_normali
save normalized_misalignment x_normalm
save normalized_resonance x_normalr
%------------------------Feature f1--------------------------------------%
for k = 1:50 
    [PSD_b(:,k),f1] = pwelch(x_normalb(:,k),[],[],[],1000); 
    [PSD_g(:,k),f1] = pwelch(x_normalg(:,k),[],[],[],1000);
    [PSD_i(:,k),f1] = pwelch(x_normali(:,k),[],[],[],1000);
    [PSD_m(:,k),f1] = pwelch(x_normalm(:,k),[],[],[],1000); 
    [PSD_r(:,k),f1] = pwelch(x_normalr(:,k),[],[],[],1000);
end 
for k1 = 1:50
    f1_b(:,k1) = (norm(PSD_b(:,k1))) / sqrt(max(size(PSD_b)));
    f1_g(:,k1) = (norm(PSD_g(:,k1))) / sqrt(max(size(PSD_g)));
    f1_i(:,k1) = (norm(PSD_i(:,k1))) / sqrt(max(size(PSD_i)));
    f1_m(:,k1) = (norm(PSD_m(:,k1))) / sqrt(max(size(PSD_m)));
    f1_r(:,k1) = (norm(PSD_r(:,k1))) / sqrt(max(size(PSD_r)));
end
%------------------(Butterworth) Feature f2-------------------------------%
[B,A] = butter(11,0.1);  %11th order low pass Butterworth Digital Filter
f2_b = filter_extract(B,A,x_normalb,Fs);
f2_g = filter_extract(B,A,x_normalg,Fs); 
f2_i = filter_extract(B,A,x_normali,Fs); 
f2_m = filter_extract(B,A,x_normalm,Fs); 
f2_r = filter_extract(B,A,x_normalr,Fs); 

%--------------------------Band pass Filter f3 (50 - 200 Hz)-------------%
[B,A] = butter(13,[0.1 0.4]); %13th Order 

f3_b = filter_extract(B,A,x_normalb,Fs);
f3_g = filter_extract(B,A,x_normalg,Fs); 
f3_i = filter_extract(B,A,x_normali,Fs); 
f3_m = filter_extract(B,A,x_normalm,Fs); 
f3_r = filter_extract(B,A,x_normalr,Fs); 

%---------------------------High pass Filter f4 (200Hz)-------------------%
[B,A] = butter(18,0.4,'high');

f4_b = filter_extract(B,A,x_normalb,Fs);
f4_g = filter_extract(B,A,x_normalg,Fs); 
f4_i = filter_extract(B,A,x_normali,Fs); 
f4_m = filter_extract(B,A,x_normalm,Fs); 
f4_r = filter_extract(B,A,x_normalr,Fs); 

%Task III
%------------------------Task III: Data Visualization--------------------%
%Desciption: Dimension reduction since there is four feature - Principal
    %component analysis (PCA) is used to map two dimensions obtained 
%Transposing since "corrcef" only works with columns
f1_b1 = transpose(f1_b); f2_b1 = transpose(f2_b);
f3_b1 = transpose(f3_b); f4_b1 = transpose(f4_b); 
f1_g1 = transpose(f1_g); f2_g1 = transpose(f2_g);
f3_g1 = transpose(f3_g); f4_g1 = transpose(f4_g);
f1_i1 = transpose(f1_i); f2_i1 = transpose(f2_i);
f3_i1 = transpose(f3_i); f4_i1 = transpose(f4_i);
f1_m1 = transpose(f1_m); f2_m1 = transpose(f2_m);
f3_m1 = transpose(f3_m); f4_m1 = transpose(f4_m);
f1_r1 = transpose(f1_r); f2_r1 = transpose(f2_r);
f3_r1 = transpose(f3_r); f4_r1 = transpose(f4_r); 

%Features Matrices
f_b = [f1_b1,f2_b1,f3_b1,f4_b1]; %Bearing fault features 
f_g = [f1_g1,f2_g1,f3_g1,f4_g1]; %Gearmesh fault features 
f_i = [f1_i1,f2_i1,f3_i1,f4_i1]; %Imbalance fault features 
f_m = [f1_m1,f2_m1,f3_m1,f4_m1]; %Misalignment fault features 
f_r = [f1_r1,f2_r1,f3_r1,f4_r1]; %Resonance fault features 

save bearing_features.mat f_b
save gearmesh_features.mat f_g
save imbalance_features.mat f_i
save misalignment_features.mat f_m
save resonance_features.mat f_r 

% Task III: Data Visualization
load bearing_features.mat 
load gearmesh_features.mat 
load imbalance_features.mat
load misalignment_features.mat 
load resonance_features.mat 

G = [f_b ; f_g ; f_i ; f_m ; f_r]; %Combining fault cases 

c = corrcoef(G); %Correlation coefficent matrix c of G 
[v,d] = eig(c); % v - EigenVector & d - EigenValues of G 
T = [v(:,end)' ; v(:,end-1)']; %Transformation matrix T from the first 2 com

z = T*G'; %Creates a 2-dimensional feature vector z 
% Scatter plot of the 2-dimensional features
figure (13)
plot(z(1,1:50), z(2,1:50),'ko') ; hold on 
plot(z(1,51:100), z(2,51:100),'bo'); hold on 
plot(z(1,101:150), z(2,101:150),'ro'); hold on 
plot(z(1,151:200), z(2,151:200),'go'); hold on 
plot(z(1,201:250), z(2,201:250),'co'); hold off
xlabel ('z1'); ylabel('z2'); 
legend({'Fault 1','Fault 2','Fault 3','Fault 4','Fault 5'},'Location',...
    'southwest','NumColumns',2)
title('PCA Feature Signal (4 Energy levels')

Fault_1 = z(:,1:50)';
Fault_2 = z(:,51:100)';
Fault_3 = z(:,101:150)';
Fault_4 = z(:,151:200)';
Fault_5 = z(:,201:250)';

% Saving - To be used for part (b) of the assignment & also for Lab A. II
save Fault_1 Fault_1  
save Fault_2 Fault_2 
save Fault_3 Fault_3 
save Fault_4 Fault_4 
save Fault_5 Fault_5
%%
%------------------------------Pattern Classification--------------------%
%-------------------------------Lab A.II---------------------------------%
%1-Nearest Neighbor Algorithm 
% Load Faults 
load Fault_1 %Bearing Fault  
load Fault_2 % Gearmesh Fault 
load Fault_3 % Imbalance Fault 
load Fault_4 % Misalignment Fault 
load Fault_5 % Resonance Fault 


%Training Data - used in construction of the classifier 
%Test Data - used to evaluate the goodness or performance of the classifier
NoOfTrainingCases = 35;  %Training cases
NoOfTestingCases = length(Fault_1) - NoOfTrainingCases; % 15 vectors
% Trainging Data (First 35 data)
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
%----------Initial Varaiables for 1-nearest neighbour search------------%
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
%----------------------------1-nearest neighbour search-----------------%
% Total number of cases
totalNoOfTestingCases = NoOfTestingCases * 5;
totalNoOfTrainingCases = NoOfTrainingCases * 5;

inferredlabels = zeros(1,totalNoOfTestingCases); 

%This loop cycles through each unlabelled item:
for unlabelledCaseIdx = 1:totalNoOfTestingCases
    unlabelledCase = testingSet(unlabelledCaseIdx, :); 
    
    %As any distance is shorter than infinity 
    shortestDistance = inf;
    shortestDistanceLabel = 0; %Assign a temporary label 
    
    %This loop cycles through each labelled item: 
    for labelledCaseIdx = 1:totalNoOfTrainingCases 
        labelledCase = trainingSet(labelledCaseIdx, :); 
        
        %Calculate the Euclidean distance: 
        currentDist = euc(unlabelledCase,labelledCase);
        
        %Check the distance 
        if currentDist < shortestDistance 
            shortestDistance = currentDist; 
            shortestDistanceLabel = trainingTarget(labelledCaseIdx);
        end 
    end %Closes the inner for loop 
    %Assign the ofund label to the vector of inferred labels: 
    inferredlabels(unlabelledCaseIdx) = shortestDistanceLabel; 
end %Outer For loop ends 

% No. of correctly classified samples 
Nc = length(find(inferredlabels == testingTarget));
% No. of all samples 
Na = length(testingTarget);

%Accuracy of Classification ACC 
Acc = 100*(Nc/Na); % Accuracy of classification (ACC) 
disp(Acc)
%%
Na=totalNoOfTestingCases;
    Nc=0;
    for i=1:Na
        if(inferredlabels(i)==testingTarget(i))
            Nc=Nc+1;
        end
    end
    Acc(i) = 100*Nc/Na;
    disp(Acc)
%-------------------------------End of Script (Lab Completed)------------%

