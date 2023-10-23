% This solves the von Neumann equation for the Frisch-Segre experiment
% S. Suleyman Kahraman, Kelvin Titimbo, Zhe He,  and Lihong V. Wang
% California Institute of Technology
% October 2022

clear all;
close all;

%%%%%%% Choose simulation parameters here. %%%%%%%%%%%%%%%%%%%
% Initial state is up for exact field, down for quad app

% Magnetic field: 1 exact, 2 quad
iB = 1;
% Interaction coefficient 1-8, 1 is the correct value, 8 is no interaction
iA = 1;
% Nuclear density matrix: 1-20
irhon = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Constants
hbar = 1.05457e-34;     % Reduced Planck constant (J s)
mu_0 = 4*pi*1e-7;       % Vacuum permeability (Tm/A)
R = 275e-12;            % van der Waals atomic radius (m) [https://periodic.lanl.gov/3.shtml]
gamma_e = -1.76e11;     % Electron gyromagnetic ratio  (1/sT). RSU = 3.0e-10
gamma_n = 1.25e7;       % Nuclear gyromagnetic ratio [gfactor*muB/hbar] (1/(Ts)) [http://triton.iqfr.csic.es/guide/eNMR/chem/]
% mu_e = -9.28e-24;     % Electron magnetic moment (J/T). RSU = 3.0e-10
% mu_n = 1.96e-27;      % Nuclear magnetic moment (J/T)  [https://www-nds.iaea.org/nuclearmoments/][https://www-nds.iaea.org/publications/indc/indc-nds-0794/]

% Spin-1/2 Pauli matrices
sigma_x = [0, 1; 1, 0];
sigma_y = [0, -1i; 1i, 0];
sigma_z = [1, 0; 0, -1];

% Spin-3/2 Pauli matrices
tau_x = [0, sqrt(3), 0, 0; sqrt(3), 0, 2, 0; 0, 2, 0, sqrt(3); 0, 0, sqrt(3), 0];
tau_y = [0, -1i*sqrt(3), 0, 0; 1i*sqrt(3), 0, -2i, 0; 0, 2i, 0, -1i*sqrt(3); 0, 0, 1i*sqrt(3), 0];
tau_z = [3, 0, 0, 0; 0, 1, 0, 0; 0, 0, -1, 0; 0, 0, 0,-3];

% 3/2 - 1/2 interaction matrix
S_ex = kron(eye(4),sigma_x);
S_nx = kron(tau_x,eye(2));
S_ey = kron(eye(4),sigma_y);
S_ny = kron(tau_y,eye(2));
S_ez = kron(eye(4),sigma_z);
S_nz = kron(tau_z,eye(2));
sigma_int = (S_ex*S_nx + S_ey*S_ny + S_ez*S_nz)/4;

% FS experimental parameters
vy = 800;               % Atom speed (m/s)
za = 1.05e-4;           % Wire position (m)
Br = 0.42e-4;           % Remnant field (T)

% Experimental data
FS_Iwire = [0.010, 0.020, 0.03, 0.05, 0.10, 0.20, 0.30, 0.5];       % in (A)
FS_data = [0.19, 6.14, 14.87, 26.68, 30.81, 26.8, 12.62, 0.1]/100;  % FS exp prob

% Wire currents to simulate 
Is = FS_Iwire;

% Converting operators to the tensor space
Sigma_z = kron(eye(4),sigma_z)/2;
Tau_z = kron(tau_z,eye(2))/2;

% % % Initial density matrix for electron
% Up state
rho_e_list{1} = [1 0; 0 0];
% Down state
rho_e_list{2} = [0 0; 0 1];

%%% Initial density matrix for nucleus
% 4 level, iso, mixed
rho_n_list{1} = eye(4)/4;
% 4 level, iso, pure
rho_n_list{2} = ones(4)/4;
% 2 level, iso, mixed
temp = ([1 0 0 1])/2;
temp = temp./ sum(temp);
rho_n_list{3} = diag(temp);
% 2 level, iso, pure
temp = ([1 0 0 1])/sqrt(2);
temp = temp ./ norm(temp);
rho_n_list{4} = temp' * temp;
% 4 level, aniso, mixed
temp = ((1+[-3:2:3]/3)/2).^2;
temp = temp./ sum(temp);
rho_n_list{5} = diag(temp);
% 4 level, aniso, pure
temp = (1+[-3:2:3]/3)/2;
temp = temp ./ norm(temp);
rho_n_list{6} = temp' * temp;
% 4 level, aniso, mixed
temp = ((1-[-3:2:3]/3)/2).^2;
temp = temp./ sum(temp);
rho_n_list{7} = diag(temp);
% 4 level, aniso, pure
temp = (1-[-3:2:3]/3)/2;
temp = temp ./ norm(temp);
rho_n_list{8} = temp' * temp;
% 4 level, aniso, mixed
temp = ((1+[-3:2:3]/3)/2);
temp = temp./ sum(temp);
rho_n_list{9} = diag(temp);
% 4 level, aniso, pure
temp = sqrt((1+[-3:2:3]/3)/2);
temp = temp ./ norm(temp);
rho_n_list{10} = temp' * temp;
% 4 level, aniso, mixed
temp = ((1-[-3:2:3]/3)/2);
temp = temp./ sum(temp);
rho_n_list{11} = diag(temp);
% 4 level, aniso, pure
temp = sqrt((1-[-3:2:3]/3)/2);
temp = temp ./ norm(temp);
rho_n_list{12} = temp' * temp;
% 2 level, aniso, mixed
rho_n_list{13} = diag([1,2,3,4]/10);
% 2 level, aniso, pure
temp = sqrt([1,2,3,4])/sqrt(10);
temp = temp ./ norm(temp);
rho_n_list{14} = temp' * temp;
% 2 level, aniso, mixed
rho_n_list{15} = diag([4,3,2,1]/10);
% 2 level, aniso, pure
temp = sqrt([4,3,2,1])/sqrt(10);
temp = temp ./ norm(temp);
rho_n_list{16} = temp' * temp;
% 4 level, aniso, mixed
temp = (1+[-3:2:3]/sqrt(15));
temp = temp./ sum(temp);
rho_n_list{17} = diag(temp);
% 4 level, aniso, pure
temp = sqrt(1+[-3:2:3]/sqrt(15));
temp = temp ./ norm(temp);
rho_n_list{18} = temp' * temp;
% 4 level, aniso, mixed
temp = (1-[-3:2:3]/sqrt(15));
temp = temp./ sum(temp);
rho_n_list{19} = diag(temp);
% 4 level, aniso, pure
temp = sqrt(1-[-3:2:3]/sqrt(15));
temp = temp ./ norm(temp);
rho_n_list{20} = temp' * temp;
% 2 level, aniso, mixed
rho_n_list{21} = diag([1/4,0,0,3/4]);
% 2 level, aniso, pure
temp = [1/2,0,0,sqrt(3)/2];
temp = temp ./ norm(temp);
rho_n_list{22} = temp' * temp;
% 2 level, aniso, mixed
rho_n_list{23} = diag([3/4,0,0,1/4]);
% 2 level, aniso, pure
temp = [sqrt(3)/2,0,0,1/2];
temp = temp ./ norm(temp);
rho_n_list{24} = temp' * temp;
% 2 level, aniso, mixed
rho_n_list{25} = diag([1/3,0,0,2/3]);
% 2 level, aniso, pure
temp = [sqrt(1/3),0,0,sqrt(2/3)];
temp = temp ./ norm(temp);
rho_n_list{26} = temp' * temp;
% 2 level, aniso, mixed
rho_n_list{27} = diag([2/3,0,0,1/3]);
% 2 level, aniso, pure
temp = [sqrt(2/3),0,0,sqrt(1/3)];
temp = temp ./ norm(temp);
rho_n_list{28} = temp' * temp;

%%% Interaction coefficients
% Experimental value at https://link.aps.org/doi/10.1103/RevModPhys.49.31
A_list(1) = 230.8598601e6 * hbar*2*pi;
% Top-hat, self-averaged 
A_list(2) = 1/2 * (-mu_0*gamma_e*gamma_n/(pi*R^3) * hbar^2);
% Gaussian, self-averaged 
A_list(3) = 16/(3*pi^2) * (-mu_0*gamma_e*gamma_n/(pi*R^3) * hbar^2);
% Hartree self-averaged 
A_list(4) = 28.4/3 * (-mu_0*gamma_e*gamma_n/(pi*R^3) * hbar^2);
% Top-hat, torque-averaged 
A_list(5) = 5/16 * (-mu_0*gamma_e*gamma_n/(pi*R^3) * hbar^2);
% Gaussian, torque-averaged 
A_list(6) = 4/(3*pi^2) * (-mu_0*gamma_e*gamma_n/(pi*R^3) * hbar^2);
% Hartree, torque-averaged 
A_list(7) = 0.138 * (-mu_0*gamma_e*gamma_n/(pi*R^3) * hbar^2);
% Electron only (Majorana) case
A_list(8) = 0;

% Map the chosen indices to the simulation parameters
A = A_list(iA);
% Chooses initial eletron state according to field
if iB == 1
    irhoe = 2;  % Start from down spin for exact field
else
    irhoe = 1;  % Start from up spin for quadrupole field
end
rho_e = rho_e_list{irhoe};
% Initial nuclear state
rho_n = rho_n_list{irhon};
% Initial density matrix in tensor space
rho_0 = kron(rho_n, rho_e);

% Time parameters
% Choose a small time step size if the Hamiltonian is large 
% (Using empirically chosen values)
if A > 10e6 * hbar*2*pi 
    dt = 1e-12; % This will be slow ~hour
else
    dt = 1e-10; % Fast ~min
end
% Total simulation time from start to end
tf = 2e-5;
% Wire location is at t=0
t = -tf/2:dt:tf/2;
       
% Parameters summary
disp('Spin 3/2 - Spin 1/2');
disp(['irhon -> ' num2str(irhon)]);
disp(['irhoe -> ' num2str(irhoe)]);
disp(['iB -> ' num2str(iB)]);
disp(['iA -> ' num2str(iA)]);

% Interaction Hamiltonian
H_int = A .* sigma_int;

% Variable initializations
pe_up = [];
Tau_z_t = [];
Sigma_z_t = [];
    
% Loop over currents
for iI = 1:length(Is)
    
    % Set current 
    I = Is(iI);
    % Initialize density matrix
    rho = rho_0;
    % Calculate the null point position in flight time
    t_NP = mu_0*I/(2*pi*Br*vy);
    
    % Loop over time
    for it = 1:length(t)
        
        % Coordinate
        y = vy*t(it);
        y_step = vy*(t(it)+dt/2);
        
        if iB == 1
            % Exact field at the current coordinates
            Bx = 0;
            By = mu_0*I*za/(2*pi*(y^2+za^2));
            Bz = Br - mu_0*I*y/(2*pi*(y^2+za^2));
            % Exact field half step after the current coordinates
            Bx_step = 0;
            By_step = mu_0*I*za/(2*pi*(y_step^2+za^2));
            Bz_step = Br - mu_0*I*y_step/(2*pi*(y_step^2+za^2));
        elseif iB == 2
            % Quadrupole field at the current coordinates
            Bx = 0;
            By = za*Br^2*2*pi/mu_0./I;
            Bz = vy*Br^2*2*pi/mu_0./I*(t(it)-t_NP);
            % Quadrupole field half step after the current coordinates
            Bx_step = 0;
            By_step = za*Br^2*2*pi/mu_0 ./ I;
            Bz_step = vy*Br^2*2*pi/mu_0./I*(t(it)+dt/2-t_NP);
        end
        
        % Hamiltonian with the external B field
        He2 = -gamma_e*hbar/2 * (sigma_x * Bx + sigma_y * By + sigma_z * Bz);
        Hn4 = -gamma_n*hbar/2 * (tau_x * Bx + tau_y * By + tau_z * Bz);
        H_e = kron(eye(4),He2);
        H_n = kron(Hn4,eye(2));
        He2_step = -gamma_e*hbar/2 * (sigma_x * Bx_step + sigma_y * By_step + sigma_z * Bz_step);
        Hn4_step = -gamma_n*hbar/2 * (tau_x * Bx_step + tau_y * By_step + tau_z * Bz_step);
        H_e_step = kron(eye(4),He2_step);
        H_n_step = kron(Hn4_step,eye(2));
        
        % Total Hamiltonian
        H = H_e + H_n + H_int;
        H_step = H_e_step + H_n_step + H_int;
        
        % Propagate density matrix (Runge-Kutta with 2 steps)
        rho_step = rho + dt/2 * (H*rho-rho*H)/(1i*hbar);
        rho = rho + dt * (H_step*rho_step-rho_step*H_step)/(1i*hbar);
        
        % Record the total probability of up electron spin
        probs = diag(rho);
        pe_up(iI,it) = sum(probs(1:2:end));
        
        % Record expectation values of z projections of the spins
        Tau_z_t(iI,it) = trace(rho*Tau_z);
        Sigma_z_t(iI,it) = trace(rho*Sigma_z);
        
        % trace(rho) should be 1 at all times. Otherwise reduce dt.        
    end
    
    % Average over the final region
    p(iI) = mean(pe_up(iI,t>tf*3/8));
end

% R-squared
SS_0 = sum(((FS_data) - mean((FS_data))).^2);
SS_1 = sum(((FS_data) - (p)).^2);
R2 = 1 - SS_1/SS_0

% Pearson correlation
rpearson = corr(p', FS_data')

% Mean squared error
mse = sum(((p)-FS_data).^2)

% String to save the results
str = ['rhon-' num2str(irhon) '_rhoe-' num2str(irhoe) '_B-' num2str(iB) '_A-' num2str(iA)];

% Make a folder to save the figures and results 
datafoldername = ['Output_', mfilename, '_', char(datetime('now','TimeZone','local','Format','yyyy-MM-dd_HH-mm-ss'))];
if ~isfolder(datafoldername )
    mkdir(datafoldername );
    disp(['Output folder ' datafoldername ' created'])
end 
    
% Plot and save the time trace for the final current value
hf = figure; Iplot = 5;
yyaxis left;  plot(vy*t/1e-3,Sigma_z_t(Iplot,:),'LineWidth',2); 
ylabel('\boldmath$\langle \sigma_{\rm{z,e}} \rangle$','FontWeight','bold'); 
ylim([-1/2 1/2]); xlim([vy*t(1) vy*t(end)]/1e-3);
yticks([-1/2:1/6:1/2]); yticklabels({'-1/2','','','0','','','1/2'});
yyaxis right; plot(vy*t/1e-3,Tau_z_t(Iplot,:),'LineWidth',2); 
ylabel('\boldmath$\langle \sigma_{\rm{z,n}} \rangle$','FontWeight','bold'); 
ylim([-3/2 3/2]); 
yticks([-3/2:1/2:3/2]); yticklabels({'-3/2','-1','-1/2','0','1/2','1','3/2'});
title(['I = ' num2str(Is(Iplot)) 'A' ],'FontSize',12,'FontWeight','normal'); 
xlabel('$y$ (mm)','FontSize',12,'FontWeight','normal'); 
grid on;
set(findall(hf,'-property','FontSize'),'FontSize',14) 
set(findall(hf,'-property','Interpreter'),'Interpreter','latex') 
print(hf,[datafoldername '/timetrace_' str '.png'],'-dpng','-painters')

% Plot and save the curve
hf = figure; 
semilogx(FS_Iwire, FS_data, 'ko', 'LineWidth', 2, 'MarkerSize', 6); hold on;
semilogx(Is,p,'rx', 'LineWidth', 2, 'MarkerSize', 8);
ylim([0 1]); yticks([0 0.5 1]); yticklabels({'0','1/2','1'});
legend('Frisch-Segre experiment','QM simulation','Box','off','Location','NorthWest'); grid on;
xlabel('Wire current (A)');
ylabel('Flip probability');
xlim([min(FS_Iwire) max(FS_Iwire)]);

set(findall(hf,'-property','FontSize'),'FontSize',14) 
set(findall(hf,'-property','Interpreter'),'Interpreter','latex') 
print(hf,[datafoldername '/curve_' str '.png'],'-dpng','-painters')

% Save the workspace
clear hf; 
save([datafoldername '/workspace_' str '.mat']);

% Save the current script
copyfile([mfilename '.m'], [datafoldername '/executedscript.m']);

