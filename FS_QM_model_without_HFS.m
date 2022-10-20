% This solves the von Neumann equation for the Frisch-Segre experiment
% S. Suleyman Kahraman, Kelvin Titimbo, Zhe He,  and Lihong V. Wang
% California Institute of Technology
% October 2022

clear all;
close all;

%%%%%%% Choose simulation parameters here. %%%%%%%%%%%%%%%%%%%
% Initial state is up for exact field, down for quad app

% Magnetic field: 1 exact, 2 quad
iB = 2;

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

% 3/2 - 1/2 interaction matrix
S_ex = sigma_x;
S_ey = sigma_y;
S_ez = sigma_z;

% FS experimental parameters
vy = 800;               % Atom speed (m/s)
za = 1.05e-4;           % Wire position (m)
Br = 0.42e-4;           % Remnant field (T)

% Experimental data
FS_Iwire = [0.010, 0.020, 0.03, 0.05, 0.10, 0.20, 0.30, 0.5];       % in (A)
FS_data = [0.19, 6.14, 14.87, 26.68, 30.81, 26.8, 12.62, 0.1]/100;  % FS exp prob

% Closed-form analytical formula from CQD
CQD_Iwire = logspace(-2, 0, 201);
crp = 0.054; crs = 0.80; crn = 48; cri = 0.57;
CQD_flipProb = exp(-(sqrt((crp./CQD_Iwire).^2 + crs^2))-crn.*CQD_Iwire.^3 - cri.*CQD_Iwire);

% Closed-form analytical formula from Majorana and Rabi
G = 2*pi./(mu_0*CQD_Iwire)*Br^2 ;
By = G*za;
W_m = exp(-pi*za*abs(gamma_e)*By/2/vy);
W_r = exp(-pi*za*abs(gamma_e/4)*By/2/vy)/4;

% Wire currents to simulate 
Is = FS_Iwire;

% Converting operators to the tensor space
Sigma_z = sigma_z/2;

% % % Initial density matrix for electron
% Up state
rho_e_list{1} = [1 0; 0 0];
% Down state
rho_e_list{2} = [0 0; 0 1];

% Map the chosen indices to the simulation parameters
% Chooses initial eletron state according to field
if iB == 1
    irhoe = 2;  % Start from down spin for exact field
else
    irhoe = 1;  % Start from up spin for quadrupole field
end
rho_e = rho_e_list{irhoe};
% Initial density matrix in tensor space
rho_0 = rho_e;

% Time parameters
% Choose a small time step size if the Hamiltonian is large 
% (Using empirically chosen value)
dt = 1e-10; 
% Total simulation time from start to end
tf = 2e-5;
% Wire location is at t=0
t = -tf/2:dt:tf/2;
       
% Parameters summary
disp('Electron only');
disp(['irhoe -> ' num2str(irhoe)]);
disp(['iB -> ' num2str(iB)]);

% Variable initializations
pe_up = [];
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
        H_e = -gamma_e*hbar/2 * (sigma_x * Bx + sigma_y * By + sigma_z * Bz);
        H_e_step = -gamma_e*hbar/2 * (sigma_x * Bx_step + sigma_y * By_step + sigma_z * Bz_step);
        
        % Total Hamiltonian
        H = H_e;
        H_step = H_e_step;
        
        % Propagate density matrix (Runge-Kutta with 2 steps)
        rho_step = rho + dt/2 * (H*rho-rho*H)/(1i*hbar);
        rho = rho + dt * (H*rho_step-rho_step*H)/(1i*hbar);
        
        % Record the total probability of up electron spin
        pe_up(iI,it) = rho(1,1);
        
        % Record expectation values of z projections of the spins
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
str = ['rhoe-' num2str(irhoe) '_B-' num2str(iB) '_' char(datetime('now','TimeZone','local','Format','yyyy-MM-dd_HH-mm-ss'))];

% Make a folder to save the figures and results 
datafoldername = ['Output_', mfilename];
if ~isfolder(datafoldername )
    mkdir(datafoldername );
    disp(['Output folder ' datafoldername ' created'])
end 
    
% Plot and save the time trace for the final current value
hf = figure; Iplot = 5;
plot(vy*t/1e-3,Sigma_z_t(Iplot,:),'LineWidth',2); 
ylabel('\boldmath$\langle \sigma_{\rm{z,e}} \rangle$','FontWeight','bold'); 
ylim([-1/2 1/2]); xlim([vy*t(1) vy*t(end)]/1e-3);
yticks([-1/2:1/6:1/2]); yticklabels({'-1/2','','','0','','','1/2'});
title(['I = ' num2str(Is(Iplot)) 'A' ],'FontSize',12,'FontWeight','normal'); 
xlabel('$y$ (mm)','FontSize',12,'FontWeight','normal'); 
grid on;
set(findall(hf,'-property','FontSize'),'FontSize',14) 
set(findall(hf,'-property','Interpreter'),'Interpreter','latex') 
print(hf,[datafoldername '/timetrace_' str '.png'],'-dpng','-painters')

% Plot and save the curve
hf = figure; 
semilogx(FS_Iwire, FS_data, 'ko', 'LineWidth', 2, 'MarkerSize', 6); hold on;
semilogx(CQD_Iwire, CQD_flipProb, '-', 'LineWidth', 1, 'Color', [0.5 0.5 0.5]);
semilogx(Is,p,'rx', 'LineWidth', 2, 'MarkerSize', 8);
ylim([0 1]); yticks([0 0.5 1]); yticklabels({'0','1/2','1'});
legend('Frisch-Segre experiment', 'CQD prediction','QM result','Box','off','Location','NorthWest'); grid on;
xlabel('Wire current (A)');
ylabel('Flip probability');
xlim([min(FS_Iwire) max(FS_Iwire)]);

set(findall(hf,'-property','FontSize'),'FontSize',14) 
set(findall(hf,'-property','Interpreter'),'Interpreter','latex') 
print(hf,[datafoldername '/curve_' str '.png'],'-dpng','-painters')

% Save the workspace
clear hf; 
save([datafoldername '/workspace_' str '.mat']);

copyfile([mfilename '.m'], [datafoldername '/' mfilename '_' str '.m']);

