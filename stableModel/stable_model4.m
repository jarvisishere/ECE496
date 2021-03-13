clear all;
patient = 1;%input('Choose Patient Data (1.SB, 2.MD, 3.TKB): ');

% choose patient ID
switch patient
    case 1
      disp('Patient ID:SB data loaded');
      hemo = load('SB_Hb.mat').SBHb;
      neut = load('SB_Neut.mat').SBNeut;
      plt = load('SB_Pit.mat').SBPit;
    case 2
       disp('Patient ID:MD data loaded');
       hemo = load('MD_Hb.mat').MDHb;
       neut = load('MD_Neut.mat').MDNeut;
       plt = load('MD_Pit.mat').MDPit;
    case 3
        disp('Patient ID:TKB data loaded')
        hemo = load('TKB_Hb.mat').TKBHb;
        neut = load('TKB_Neut.mat').TKBNeut;
        plt = load('TKB_Pit.mat').TKBPit;

end

treatment = load('SB_treat.mat').SBTreat;

% testingMode = input('Choose testing mode (1.Functinality, 2.Input date): ');
% switch testingMode
%     case 1
        i = 18; %time point at which the training starts
        j = 53 - i; % # of data poitns used for training
        % Use the data (and information about the treatment protocol) 
        % between cycle 1 and cycle 2 to train the model. (Ignore the 
        % data prior to the start of cycle 1)
% end
x = [neut plt]';
ceil = max(x,[],2) * 1.5; % Ceil: max measured value * 1.5
u = treatment';
len = length(x);

X_dim = 3;
X = [x(:,1:len-2);x(:,2:len-1);x(:,3:len)];
U = [u(:,1:len-2);u(:,2:len-1);u(:,3:len)];
ceil = [ceil;ceil;ceil];

% X_dim = 2;
% X = [x(:,1:len-1);x(:,2:len)];
% U = [u(:,1:len-1);u(:,2:len)];
% ceil = [ceil;ceil];

X0 = X(:,i:i+j);
X1 = X(:,i+1:i+j+1);
U0 = U(:,i:i+j);

sigma_XU = [X0*X0' X0*U0';U0*X0' U0*U0'];
AB_hat = X1*([X0' U0'])*(sigma_XU^(-1));
A_hat = AB_hat(:,1:X_dim*2);
B_hat = AB_hat(:,X_dim*2+1:X_dim*2+X_dim*2);

e = X1 - (A_hat*X0 + B_hat*U0);
e_avg = sum(e,2)/j; %sample mean
Q_hat = ((e-e_avg)*(e-e_avg)')/(j-1); %sample covariance
W = Q_hat*Q_hat';
We = [W zeros(X_dim*2,X_dim*2);zeros(X_dim*2,X_dim*2) zeros(X_dim*2,X_dim*2)];

[Q,R] = qr([U0' X0']);
R22 = R(X_dim*2+1:X_dim*2+X_dim*2,X_dim*2+1:X_dim*2+X_dim*2);

gamma = 0.99;

c = (max(svd(inv(R22')*A_hat*R22'))/gamma - 1)/min(svd(inv(R22')*W*inv(R22))); %#ok<*MINV>

AB_tilde = [X1*X0' X1*U0']*(sigma_XU+c*We)^(-1);
A_tilde = AB_tilde(:,1:X_dim*2);
B_tilde = AB_tilde(:,X_dim*2+1:X_dim*2+X_dim*2);
outputLength = 7; % predict next 7 days
esti = X(:,1:i+j);
for a = i+j+1:i+j+outputLength
    esti(:,a) = A_tilde*esti(:,a-1)+B_tilde*U(:,a-1);
    esti(:,a) = max(esti(:,a),zeros(X_dim*2,1)); % Floor: 0
    esti(:,a) = min(esti(:,a),ceil);% Ceil: max measured value * 1.5
end

EstError = [0, 0, 0 ];
errorCal1 = zeros(outputLength,1);
denominator = 0;
figure;
plot(hemo(1:end));
hold on
plot(neut(1:end));
hold on
plot(plt(1:end));
legend('hemo',', neut', 'plt ');

%plot data
figure;
subplot(3,2,1)
plot(esti(1,1:i+j+outputLength))
hold on
plot(hemo(1:i+j+outputLength))
switch patient
    case 1
        title("Patient ID: SB Hemoglobin Data")
        xline(21,'r--')
        xline(54,'r--')
    case 2
        title("Patient ID: MD Hemoglobin Data")
        xline(17,'r--')
        xline(55,'r--')
    case 3
        title("Patient ID: TKB Hemoglobin Data")
        xline(20,'r--')
        xline(70,'r--')
end
ylabel('Hemoglobin (g/L)'); 
xlabel('days');
legend('estimated data','actual data')
xlim([50 60])
subplot(3,2,2)
% calculate estimation error
diff = hemo(1:i+j+outputLength)- esti(1,:)';
plot(abs(diff))
title("Hemoglobin Estimation Error")
ylabel('Hemo Difference (g/L)'); 
xlabel('days');
legend('estimated error');
xlim([50 60])
for a = 1:i+j+outputLength 
    errorCal1(a) = abs(hemo(a)- esti(1,a)') / (sum(hemo(1:i+j+outputLength))/(i+j+outputLength));
end
EstError(1) = var(errorCal1);
errorCal2 = zeros(i+j+outputLength,1);

subplot(3,2,3);
plot(esti(2,1:i+j+outputLength))
hold on
plot(neut(1:i+j+outputLength))
switch patient
    case 1
        title("Patient ID: SB Neutrophils Data")
        xline(21,'r--')
        xline(54,'r--')
    case 2
        title("Patient ID: MD Neutrophils Data")
        xline(17,'r--')
        xline(55,'r--')
    case 3
        title("Patient ID: TKB Neutrophils Data")
        xline(20,'r--')
        xline(70,'r--')
end
ylabel('Neutrophils *10^9(cells/L)'); 
xlabel('days');
legend('estimated data','actual data')
xlim([50 60])
subplot(3,2,4)
% calculate estimation error
diff = neut(1:i+j+outputLength)- esti(2,:)';
plot(abs(diff))
title("Neutrophils Estimation Error")
ylabel('Neutrophils *10^9(cells/L)'); 
xlabel('days');
legend('estimated error');
xlim([50 60])
for a = 1:i+j+outputLength 
    
    errorCal2(a) = abs(neut(a)- esti(2,a)') / (sum(neut(1:i+j+outputLength)) / (i+j+outputLength));
   
end
 EstError(2) = var(errorCal2);
 errorCal3 = zeros(i+j+outputLength,1);

subplot(3,2,5);
plot(esti(3,1:i+j+outputLength))
hold on
plot(plt(1:i+j+outputLength))
switch patient
    case 1
        title("Patient ID: SB Platelets Data")
        xline(21,'r--')
        xline(54,'r--')
    case 2
        title("Patient ID: MD Platelets Data")
        xline(17,'r--')
        xline(55,'r--')
    case 3
        title("Patient ID: TKB Platelets Data")
        xline(20,'r--')
        xline(70,'r--')
end
ylabel('Platelets *10^9(cells/L)');
xlabel('days');
legend('estimated data','actual data')
xlim([50 60])
% calculate estimation error
subplot(3,2,6)
diff = plt(1:i+j+outputLength)- esti(3,:)';
plot(abs(diff))
title("Platelets Estimation Error")
ylabel('Platelets *10^9(cells/L)'); 
xlabel('days');
legend('estimated error');
xlim([50 60])
for a = 1:i+j+outputLength 
      errorCal3(a) = abs(plt(a)- esti(3,a)') /(sum(plt(1:i+j+outputLength))/(i+j+outputLength));
end
EstError(3) = var(errorCal3);


 



