clear all;
for patient = 1:3
% patient = 1;%input('Choose Patient Data (1.SB, 2.MD, 3.TKB): ');

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

switch patient
    case 1
        treatment = load('SB_treat.mat').SBTreat;
        hemo = load('SB_Hb.mat').SBHb;
    case 2
        treatment = load('MD_treat.mat').MDTreat;
        hemo = load('MD_Hb.mat').MDHb;
    case 3
        treatment = load('TKB_treat.mat').TKBTreat;
        hemo = load('TKB_Hb.mat').TKBHb;
end

switch patient
    case 1
        i = 18; %time point at which the training starts
        j = 53 - i; % # of data poitns used for training
    case 2
        i = 14;
        j = 54-i;
    case 3
        i = 17;
        j = 69-i;
end
% testingMode = input('Choose testing mode (1.Functinality, 2.Input date): ');
% switch testingMode
%     case 1
%         i = 18; %time point at which the training starts
%         j = 53 - i; % # of data poitns used for training
        % Use the data (and information about the treatment protocol) 
        % between cycle 1 and cycle 2 to train the model. (Ignore the 
        % data prior to the start of cycle 1)
% end
% x = [neut plt]';
x = [plt]';
x_dim = 1; % # of blood count parameters measured in 1 day
ceil = max(x,[],2) * 1.5; % Ceil: max measured value * 1.5
u = [treatment hemo neut]';
u_dim = 4; % # of treatment input (and hemo) measured in 1 day
% u = treatment';
len = length(x);

X_dim = 1; % # of consecutive days for CBC measurement
% X = [x(:,1:len-2);x(:,2:len-1);x(:,3:len)];
X = [x(:,1:len)];
U_dim = 1; % # of consecutive days for input treatment measurement
% U = [u(:,1:len-2);u(:,2:len-1);u(:,3:len)];
% U = [u(:,1:len-1);u(:,2:len)];
U = [u(:,1:len)];


% ceil = [ceil;ceil;ceil];
ceil = [ceil];

% X_dim = 2;
% X = [x(:,1:len-1);x(:,2:len)];
% U_dim = 2;
% U = [u(:,1:len-1);u(:,2:len)];
% ceil = [ceil;ceil];

X0 = X(:,i:i+j);
X1 = X(:,i+1:i+j+1);
U0 = U(:,i:i+j);

sigma_XU = [X0*X0' X0*U0';U0*X0' U0*U0'];
AB_hat = X1*([X0' U0'])*(sigma_XU^(-1));
A_hat = AB_hat(:,1:X_dim*x_dim);
B_hat = AB_hat(:,X_dim*x_dim+1:X_dim*x_dim+U_dim*u_dim);

e = X1 - (A_hat*X0 + B_hat*U0);
e_avg = sum(e,2)/j; %sample mean
Q_hat = ((e-e_avg)*(e-e_avg)')/(j-1); %sample covariance
W = Q_hat*Q_hat';
We = [W zeros(X_dim*x_dim,U_dim*u_dim);zeros(U_dim*u_dim,X_dim*x_dim) zeros(U_dim*u_dim,U_dim*u_dim)];

[Q,R] = qr([U0' X0']);
R22 = R(U_dim*u_dim+1:U_dim*u_dim+X_dim*x_dim,U_dim*u_dim+1:U_dim*u_dim+X_dim*x_dim);

gamma = 0.99;

c = (max(svd(inv(R22')*A_hat*R22'))/gamma - 1)/min(svd(inv(R22')*W*inv(R22))); %#ok<*MINV>

AB_tilde = [X1*X0' X1*U0']*(sigma_XU+c*We)^(-1);
A_tilde = AB_tilde(:,1:X_dim*x_dim);
B_tilde = AB_tilde(:,X_dim*x_dim+1:X_dim*x_dim+U_dim*u_dim);
outputLength = 7; % predict next 7 days
esti = X(:,1:i+j);
for a = i+j+1:i+j+outputLength
    esti(:,a) = A_tilde*esti(:,a-1)+B_tilde*U(:,a-1);
    esti(:,a) = max(esti(:,a),zeros(X_dim*x_dim,1)); % Floor: 0
    esti(:,a) = min(esti(:,a),ceil);% Ceil: max measured value * 1.5
end
% 
% EstError = [ 0, 0 ];
% errorCal1 = zeros(outputLength,1);
% denominator = 0;
% %{
%  figure;
%   plot(neut(1:end));
%  switch patient
%      case 1
%          title("Patient ID: SB ")
%          xline(21,'r--')
%          xline(54,'r--')
%           legend('neut','treatment 1', 'treatment 2');
%      case 2
%          title("Patient ID: MD ")
%            xline(17,'r--')
%          xline(55,'r--')
%           legend('neut', 'treatment 1', 'treatment 2');
%      case 3
%          title("Patient ID: TKB")
%          xline(20,'r--')
%          xline(70,'r--')
%           legend('neut', 'treatment 1', 'treatment 2');
%   end
%  figure;
%   plot(plt(1:end));
%  switch patient
%      case 1
%          title("Patient ID: SB ")
%          xline(21,'r--')
%          xline(54,'r--')
%           legend( 'plt','treatment 1', 'treatment 2');
%      case 2
%          title("Patient ID: MD ")
%            xline(17,'r--')
%          xline(55,'r--')
%           legend( 'plt','treatment 1', 'treatment 2');
%      case 3
%          title("Patient ID: TKB")
%          xline(20,'r--')
%          xline(70,'r--')
%           legend( 'plt','treatment 1', 'treatment 2');
%   end
% %}
% 
% %plot data
% % figure;
% % subplot(3,2,1)
% % plot(esti(1,1:i+j+outputLength))
% % hold on
% % plot(hemo(1:i+j+outputLength))
% % switch patient
% %     case 1
% %         title("Patient ID: SB Hemoglobin Data")
% %         xline(21,'r--')
% %         xline(54,'r--')
% %     case 2
% %         title("Patient ID: MD Hemoglobin Data")
% %         xline(17,'r--')
% %         xline(55,'r--')
% %     case 3
% %         title("Patient ID: TKB Hemoglobin Data")
% %         xline(20,'r--')
% %         xline(70,'r--')
% % end
% % ylabel('Hemoglobin (g/L)'); 
% % xlabel('days');
% % legend('estimated data','actual data')
% % % xlim([50 60])
% % subplot(3,2,2)
% % % calculate estimation error
% % diff = hemo(1:i+j+outputLength)- esti(1,:)';
% % plot(abs(diff))
% % title("Hemoglobin Estimation Error")
% % ylabel('Hemo Difference (g/L)'); 
% % xlabel('days');
% % legend('estimated error');
% % % xlim([50 60])
% % for a = 1:i+j+outputLength 
% %     errorCal1(a) = abs(hemo(a)- esti(1,a)') / (sum(hemo(1:i+j+outputLength))/(i+j+outputLength));
% % end
% % EstError(1) = var(errorCal1);
% errorCal2 = zeros(i+j+outputLength,1);
% figure;
% subplot(2,2,1);
% plot(esti(1,1:i+j+outputLength))
% hold on
% plot(neut(1:i+j+outputLength))
% switch patient
%     case 1
%         title("Patient ID: SB Neutrophils Data")
%         xline(21,'r--')
%         xline(54,'r--')
%     case 2
%         title("Patient ID: MD Neutrophils Data")
%         xline(17,'r--')
%         xline(55,'r--')
%     case 3
%         title("Patient ID: TKB Neutrophils Data")
%         xline(20,'r--')
%         xline(70,'r--')
% end
% ylabel('Neutrophils *10^9(cells/L)'); 
% xlabel('days');
% legend('estimated data','actual data')
% switch patient
%     case 1
%         xlim([50 60])
%     case 2
%         xlim([50 60])
%     case 3
%         xlim([60 80])
% end
% subplot(2,2,2)
% % calculate estimation error
% diff = neut(1:i+j+outputLength)- esti(1,:)';
% plot(abs(diff))
% title("Neutrophils Estimation Error")
% ylabel('Neutrophils *10^9(cells/L)'); 
% xlabel('days');
% legend('estimated error');
% switch patient
%     case 1
%         xlim([50 60])
%     case 2
%         xlim([50 60])
%     case 3
%         xlim([65 75])
% end
% for a = 1:i+j+outputLength 
%     
%     errorCal2(a) = abs(neut(a)- esti(2,a)') / (sum(neut(1:i+j+outputLength)) / (i+j+outputLength));
%    
% end
%  EstError(1) = var(errorCal2);
%  errorCal3 = zeros(i+j+outputLength,1);
% 
% subplot(2,2,3);
% plot(esti(2,1:i+j+outputLength))
% hold on
% plot(plt(1:i+j+outputLength))
% switch patient
%     case 1
%         title("Patient ID: SB Platelets Data")
%         xline(21,'r--')
%         xline(54,'r--')
%     case 2
%         title("Patient ID: MD Platelets Data")
%         xline(17,'r--')
%         xline(55,'r--')
%     case 3
%         title("Patient ID: TKB Platelets Data")
%         xline(20,'r--')
%         xline(70,'r--')
% end
% ylabel('Platelets *10^9(cells/L)');
% xlabel('days');
% legend('estimated data','actual data')
% switch patient
%     case 1
%         xlim([50 60])
%     case 2
%         xlim([50 60])
%     case 3
%         xlim([60 80])
% end
% % calculate estimation error
% subplot(2,2,4)
% diff = plt(1:i+j+outputLength)- esti(2,:)';
% plot(abs(diff))
% title("Platelets Estimation Error")
% ylabel('Platelets *10^9(cells/L)'); 
% xlabel('days');
% legend('estimated error');
% switch patient
%     case 1
%         xlim([50 60])
%     case 2
%         xlim([50 60])
%     case 3
%         xlim([65 75])
% end
% for a = 1:i+j+outputLength 
%       errorCal3(a) = abs(plt(a)- esti(3,a)') /(sum(plt(1:i+j+outputLength))/(i+j+outputLength));
% end
% EstError(2) = var(errorCal3);
% 
figure
plot(esti(1,:));
hold on;
plot(plt);
switch patient
    case 1
        xlim([50 60])
    case 2
        xlim([50 60])
    case 3
        xlim([60 80])
end

end
%  
% 
% 
% 


