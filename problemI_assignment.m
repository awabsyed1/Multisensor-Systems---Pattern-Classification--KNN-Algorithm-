%ACS 6124: Multisensor Assignment I 
%Author : Awabullah Syed 
%Date  : 10th April 2021 
%------------------------------Problem I(Assignment)----------------------%
%------------------------Problem I (Part A)-------------------------------%
% Data
clear; clc;
load bearing.mat %Vibration MEAS of bearing-defect (Ts = 50sec,fs = 1 000)
load gearmesh.mat %Vibration MEAS of gearmesh rig (Ts = 50sec, fs = 1 000)
load misalignment.mat % MEAS of misalignment rig (Ts = 50sec, fs = 1 000)
load imbalance.mat % MEAS of imbalance rig (Ts = 50sec, fs = 1 000)
load resonance.mat % MEAS of resonance rig (Ts = 50sec, fs = 1 000)
% Normalized 
load normalized_bearing 
load normalized_gearmesh 
load normalized_imbalance 
load normalized_misalignment 
load normalized_resonance 
normal = [x_normalb;x_normalg;x_normali;x_normalm;x_normalr];
%----------------------------Part A---------------------------------------%
%Pre allocation for speed 
f = 1000;

%---------------------------Feature f1-----------------------------------%
%Feature, f1 - Low-pass filter (25Hz - 25Hz / fs/2 - Wn = 0.05) / 7th order
[B,A] = butter(7,0.05,'low');  %7th order low pass Butterworth Digital Filter
for k = 1:50 %Applying low pass 7th order filter 
    y1_b(:,k) = filter(B,A,x_normalb(:,k));
    y1_g(:,k) = filter(B,A,x_normalg(:,k)); 
    y1_i(:,k)= filter(B,A,x_normali(:,k));
    y1_m(:,k) = filter(B,A,x_normalm(:,k));
    y1_r(:,k) = filter(B,A,x_normalr(:,k));
end
for k2 = 1:50 %Calculating Power Spectral Density (PSD) using Welchs method
    PSD_low_b(:,k2) = pwelch(y1_b(:,k2),[],[],[],1000);
    PSD_low_g(:,k2) = pwelch(y1_g(:,k2),[],[],[],1000);
    PSD_low_i(:,k2) = pwelch(y1_i(:,k2),[],[],[],1000);
    PSD_low_m(:,k2) = pwelch(y1_m(:,k2),[],[],[],1000);
    PSD_low_r(:,k2) = pwelch(y1_r(:,k2),[],[],[],1000);
end
for k3 = 1:50 % Feature, f1
f1_b(:,k3) = (norm(PSD_low_b(:,k3))) / sqrt(max(size(PSD_low_b)));
f1_g(:,k3) = (norm(PSD_low_g(:,k3))) / sqrt(max(size(PSD_low_g)));
f1_i(:,k3) = (norm(PSD_low_i(:,k3))) / sqrt(max(size(PSD_low_i)));
f1_m(:,k3) = (norm(PSD_low_m(:,k3))) / sqrt(max(size(PSD_low_m)));
f1_r(:,k3) = (norm(PSD_low_r(:,k3))) / sqrt(max(size(PSD_low_r)));
end
%----------------------------Feature f2-----------------------------------%
% Feature, f2 - Band-Pass filter (25 -50) [0.05 0.1]
[B,A] = butter(6,[0.05 0.1]); %6th Order, [25Hz - 50Hz]
f2_b = filter_extract(B,A,x_normalb,f);
f2_g = filter_extract(B,A,x_normalg,f); 
f2_i = filter_extract(B,A,x_normali,f); 
f2_m = filter_extract(B,A,x_normalm,f); 
f2_r = filter_extract(B,A,x_normalr,f); 

%------------------------Feature f3---------------------------------------%
% Feature, f3 - Band-Pass filter (50 - 100Hz) [0.1 0.2] / 9th Order 
[B,A] = butter(9,[0.1 0.2]); %9th Order, [25Hz - 50Hz]

f3_b = filter_extract(B,A,x_normalb,f);
f3_g = filter_extract(B,A,x_normalg,f); 
f3_i = filter_extract(B,A,x_normali,f); 
f3_m = filter_extract(B,A,x_normalm,f); 
f3_r = filter_extract(B,A,x_normalr,f);

%-------------------------Feature, f4-------------------------------------% 
[B,A] = butter(8,[0.2 0.4]); %8th Order, [100 - 200 Hz]

f4_b = filter_extract(B,A,x_normalb,f);
f4_g = filter_extract(B,A,x_normalg,f); 
f4_i = filter_extract(B,A,x_normali,f); 
f4_m = filter_extract(B,A,x_normalm,f); 
f4_r = filter_extract(B,A,x_normalr,f);

%--------------------------Feature f5-------------------------------------%
% Feature, f5 
[B,A] = butter(9,[0.4 0.7]); %9th Order, [200 - 350 Hz]
f5_b = filter_extract(B,A,x_normalb,f);
f5_g = filter_extract(B,A,x_normalg,f); 
f5_i = filter_extract(B,A,x_normali,f); 
f5_m = filter_extract(B,A,x_normalm,f); 
f5_r = filter_extract(B,A,x_normalr,f);

%---------------------------Feature f6-----------------------------------%
% Feature, f6 
[B,A] = butter(16,0.7,'high'); %16th Order, [200 - 350 Hz]
f6_b = filter_extract(B,A,x_normalb,f);
f6_g = filter_extract(B,A,x_normalg,f); 
f6_i = filter_extract(B,A,x_normali,f); 
f6_m = filter_extract(B,A,x_normalm,f); 
f6_r = filter_extract(B,A,x_normalr,f);

%--------------PCA Analysis (Dimension Reduction--------------------------%
% Principle Component Analysis (PCA)
%Transposing since "corrcef" only works with columns
f1_b1 = transpose(f1_b); f2_b1 = transpose(f2_b);
f3_b1 = transpose(f3_b); f4_b1 = transpose(f4_b); 
f5_b1 = transpose(f5_b); f6_b1 = transpose(f6_b);

f1_g1 = transpose(f1_g); f2_g1 = transpose(f2_g);
f3_g1 = transpose(f3_g); f4_g1 = transpose(f4_g);
f5_g1 = transpose(f5_g); f6_g1 = transpose(f6_g);

f1_i1 = transpose(f1_i); f2_i1 = transpose(f2_i);
f3_i1 = transpose(f3_i); f4_i1 = transpose(f4_i);
f5_i1 = transpose(f5_i); f6_i1 = transpose(f6_i);

f1_m1 = transpose(f1_m); f2_m1 = transpose(f2_m);
f3_m1 = transpose(f3_m); f4_m1 = transpose(f4_m);
f5_m1 = transpose(f5_m); f6_m1 = transpose(f6_m);

f1_r1 = transpose(f1_r); f2_r1 = transpose(f2_r);
f3_r1 = transpose(f3_r); f4_r1 = transpose(f4_r); 
f5_r1 = transpose(f5_r); f6_r1 = transpose(f6_r);

%Features Matrices 
f_b = [f1_b1,f2_b1,f3_b1,f4_b1,f5_b1,f6_b1]; %Bearing fault features 
f_g = [f1_g1,f2_g1,f3_g1,f4_g1,f5_g1,f6_g1]; %Gearmesh fault features 
f_i = [f1_i1,f2_i1,f3_i1,f4_i1,f5_i1,f6_i1]; %Imbalance fault features 
f_m = [f1_m1,f2_m1,f3_m1,f4_m1,f5_m1,f6_m1]; %Misalignment fault features 
f_r = [f1_r1,f2_r1,f3_r1,f4_r1,f5_r1,f6_r1]; %Resonance fault features 

G = [f_b ; f_g ; f_i ; f_m ; f_r]; %Combining fault cases 

c = corrcoef(G); %Correlation coefficent matrix c of G 
[v,d] = eig(c); % v - EigenVector & d - EigenValues of G 
T = [v(:,end)' ; v(:,end-1)']; %Transformation matrix T from the first 2 com

z = T*G'; %Creates a 2-dimensional feature vector z 
% Scatter plot of the 2-dimensional features

%-----------------------------Plot--------------------------------------%
figure (1)
%plot(z(1,:),z(2,:),'o')
plot(z(1,1:50), z(2,1:50),'ko') ; hold on 
plot(z(1,51:100), z(2,51:100),'bo'); hold on 
plot(z(1,101:150), z(2,101:150),'ro'); hold on 
plot(z(1,151:200), z(2,151:200),'go'); hold on 
plot(z(1,201:250), z(2,201:250),'co'); hold off
xlabel ('z1'); ylabel('z2'); 
legend({'Fault 1','Fault 2','Fault 3','Fault 4','Fault 5'},'Location',...
    'northeast','NumColumns',2)
title('PCA Feature Signal (6 Energy levels)')