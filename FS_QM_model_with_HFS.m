% This solves the von Neumann equation for the Frisch-Segre experiment
% S. Suleyman Kahraman, Kelvin Titimbo, Zhe He,  and Lihong V. Wang
% California Institute of Technology
% March 2024

clear all;
close all;

global gamma_e gamma_n sigma_z sigma_y sigma_x tau_z tau_y tau_x H_int ... 
    n_I hbar mu_0 Br v za field;

%%%%%%% Choose simulation parameters here. %%%%%%%%%%%%%%%%%%%
% Initial state is up for exact field, down for quad app

% Magnetic field: 1 exact field, 2 quadrupole approximation
field = 1;
% Initial electron spin state: 1 ms=-1/2, 2 ms=+1/2
ms_initial = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Constants
hbar = 1.05457e-34;     % Reduced Planck constant (J s)
mu_0 = 4*pi*1e-7;       % Vacuum permeability (Tm/A)
gamma_e = -1.76e11;     % Electron gyromagnetic ratio  (1/sT). RSU = 3.0e-10

% 39K constants
gamma_n = 1.25e7;       % Nuclear gyromagnetic ratio [gfactor*muB/hbar] (1/(Ts)) [http://triton.iqfr.csic.es/guide/eNMR/chem/]
spin_I = 3/2;
a_hfs = 230.8598601e6;  % Experimental value at https://link.aps.org/doi/10.1103/RevModPhys.49.31

% FS experimental parameters
v = 800;                % Atom speed (m/s)
za = 1.05e-4;           % Wire position (m)
Br = 0.42e-4;           % Remnant field (T)
L_IR = 17.6e-3;

% Experimental data
FS_Iwire = [0.010, 0.020, 0.03, 0.05, 0.10, 0.20, 0.30, 0.5];       % in (A)
FS_data = [0.19, 6.14, 14.87, 26.68, 30.81, 26.8, 12.62, 0.1]/100;  % FS exp prob

% Wire currents to simulate 
Iwire_list = FS_Iwire;

% Spin-1/2 Pauli matrices
sigma_x = [0, 1; 1, 0];
sigma_y = [0, -1i; 1i, 0];
sigma_z = [1, 0; 0, -1];

% Nuclear Hilbert space dimension 
n_I = 2*spin_I + 1;
% Nuclear Pauli matrices
sm = spin_I:-1:-spin_I;
tau_z = 2*diag(sm);
tauplus = diag(sqrt((spin_I - sm(2:end)).*(spin_I + sm(2:end) + 1)),1);
tauminus = diag(sqrt((spin_I + sm(1:end-1)).*(spin_I - sm(1:end-1) + 1)),-1);
tau_x = 2*(tauplus+tauminus)/2;
tau_y = 2*(-1i*tauplus+1i*tauminus)/2;

% 3/2 - 1/2 interaction matrix
S_ex = kron(eye(4),sigma_x);
S_nx = kron(tau_x,eye(2));
S_ey = kron(eye(4),sigma_y);
S_ny = kron(tau_y,eye(2));
S_ez = kron(eye(4),sigma_z);
S_nz = kron(tau_z,eye(2));
sigma_int = (S_ex*S_nx + S_ey*S_ny + S_ez*S_nz)/4;
A_hfs = a_hfs * hbar*2*pi;
H_int = A_hfs * sigma_int;

% Converting operators to the tensor space
Sigma_z = kron(eye(4),sigma_z)/2;
Tau_z = kron(tau_z,eye(2))/2;

% Initial nuclear spin state mixture
rho_n_i = ones(n_I,1)/n_I;      % Maximally mixed
rho_i = zeros(2*n_I,2*n_I);
% Initial electron spin state
if ms_initial == 1
    rho_e_i = [1; 0];                   % Down spin, ms = -1/2
    % Combined initial state
    rho_i(1:n_I,1:n_I) = diag(rho_n_i(end:-1:1));
else 
    rho_e_i = [0; 1];                   % Up spin, ms = +1/2
    % Combined initial state
    rho_i(n_I+1:2*n_I,n_I+1:2*n_I) = diag(rho_n_i);
end

% Flight time
tmax = +L_IR/v/2;  % final time
tmin = -L_IR/v/2;  % initial time
tspan = [tmin, tmax];

% Loop over currents
for iI = 1:length(Iwire_list)
    % Print current iteration
    disp(['I_w = ' num2str(Iwire_list(iI))]);

    % Instantaneous eigenstates at the entrance
    Ui = instantaneousEigStates(Iwire_list(iI), tmin);

    % Set initial density matrix
    rho0 = Ui * rho_i * Ui';
    u0 = zeros(2*n_I,4*n_I);
    u0(:,1:2*n_I) = real(rho0);
    u0(:,2*n_I+1:4*n_I) = imag(rho0);
    
    % Solve the ODE problem
    % Other solvers such as ode23, ode45, 
    [t,ut] = ode15s(@(t,u) VonNeumann(u, Iwire_list(iI), t), tspan, u0);
    
    % Reshape variables from the solver into matrix
    ut = reshape(ut,[size(ut,1) 2*n_I 4*n_I]);
    Nt = size(ut,1);
    
    % Variable initialization
    pe_flip_t = zeros(Nt,1);
    p_inst = zeros(2*n_I,Nt);
    rhos_inst = zeros(2*n_I,2*n_I,Nt);

    % Calculate the relevant variables
    for it = 1:Nt
        % Complex density matrix
        rhototal = squeeze(ut(it,1:2*n_I,1:2*n_I) + 1i * ut(it,1:2*n_I,(1+2*n_I):4*n_I));
    
        % Density matrix in the instantaneous eigenstate basis
        Ut = instantaneousEigStates(Iwire_list(iI), t(it));
        rhos_inst(:,:,it) = Ut' * rhototal * Ut;

        % Measurement probabilities of each state
        p_inst(:,it) = real(diag(rhos_inst(:,:,it)));

        % Electron spin flip probability
        if ms_initial == 1
            pe_flip_t(it,1) = sum(p_inst((1+n_I):2*n_I, it));
        else
            pe_flip_t(it,1) = sum(p_inst((1):n_I, it));
        end

    end
    % Final flip probability 
    pe_flip(iI,1) = real(pe_flip_t(end,1));

    % Plot the evolution of the diagonal of the density matrix for a particular current
    if iI == 5
        figure;
        plot(t*1e6, p_inst, 'LineWidth', 1.5)
        xlabel('Time [us]')
        ylabel('Population')
        legend('Lowest energy state', '.', '.', '.', '.', '.', '.', 'Highest energy state', 'Location', 'West')
        xlim(tspan.*1e6);
    end

end

% R-squared
SS_0 = sum((FS_data - mean(FS_data)).^2);
SS_1 = sum((FS_data - (pe_flip)').^2);
R2 = 1 - SS_1./SS_0;
disp(['R^2 = ' num2str(round(1000*R2)/1000)])

% Analytical curve data
Iws = 10 .^(linspace(-2,0,201));
% delta = za*abs(gamma_e)*( 2*pi./(mu_0*Iws)*Br^2 *za)/2/v;
Bx = 0; By = 0; Bz = 0; 
delta = abs(gamma_e)/(2*v) * (By*za + ...
    (By^2 + (Bz+Br)^2) * pi*za^2 ./ (mu_0*Iws) +...
    (Bx^2+By^2)/(By^2+(Bz+Br)^2) * mu_0*Iws ./ (4*pi) );
yNP = v * mu_0 * Iwire_list * (Br) / ( 2 * pi * v * ( (Br )^2) );
% Majorana solution
W_m = exp(-2*pi*delta);
alpha = 2 * asin(exp(-pi*delta/4));
W_r = rho_n_i(4)*4*sin(alpha/2).^2.*cos(alpha/2).^6 + ...
      rho_n_i(3)*6*sin(alpha/2).^4.*cos(alpha/2).^4 + ...
      rho_n_i(2)*4*sin(alpha/2).^6.*cos(alpha/2).^2 + ...
      rho_n_i(1)*sin(alpha/2).^8;

% Plot the curve
figure;
hold on;
scatter(Iwire_list, FS_data, 'ko', 'LineWidth', 2); 
scatter(Iwire_list, pe_flip, 72, 'rx', 'LineWidth', 2) 
plot(Iws, W_m, '--');
plot(Iws, W_r, 'b-');
xlabel('Wire current [A]')
ylabel('Flip probability')
title(['R^2=' num2str(round(1000*R2)/1000) ' Br=' num2str(round(Br/1e-6)) 'uT v=' num2str(round(v)) 'm/s za=' num2str(round(za/1e-6)) 'um'])
set(gca,'XScale','log')
legend('FS data','Simulation','Majorana','Rabi','Location','NorthWest')



function [Bt] = FSField(Iw, t)
    global mu_0  Br v za field;
    
    if field == 2
        % Null point
        t_NP = mu_0 * Iw / ( 2 * pi * Br * v);
        % Quadrupole field:
        Byt = za * Br^2 * 2 * pi / (mu_0 * Iw);
        Bzt = v * (t - t_NP) * Br^2 * 2 * pi / (mu_0 * Iw);
        Bxt = 0 ;
    else
        % Exact field:
        Byt = mu_0 * Iw / (2 * pi * (za^2 + (v * t)^2)) * za;
        Bzt = Br - mu_0 *Iw / (2 * pi * (za^2 + (v * t)^2)) * v * t;
        Bxt = 0 ;
    end
    Bt = [Bxt;Byt;Bzt];
end


% Defnining the differential equation for the solver
function du = VonNeumann(u, Iw, t)
    % u[:,1:2*n_I ]: real(kron(rhon,rhoe))
    % u[:,2*n_I+1:4*n_I]: imag(kron(rhon,rhoe))
    
    global hbar n_I;
    
    Bt = FSField(Iw,t);

    H = Hamiltonian(Bt);

    u = reshape(u,[2*n_I,4*n_I]);
    
    % von Neumann equation
    % du = (H*u - u*H) ./ (1i*hbar);
    term1real = real(H) * u(:,1:2*n_I ) - imag(H) * u(:,2*n_I+1:4*n_I);
    term2real = u(:,1:2*n_I ) * real(H) - u(:,2*n_I+1:4*n_I) * imag(H);
    term1comp = imag(H) * u(:,1:2*n_I ) + real(H) * u(:,2*n_I+1:4*n_I);
    term2comp = u(:,2*n_I+1:4*n_I) * real(H) + u(:,1:2*n_I ) * imag(H);
    totalreal =  (term1comp - term2comp) ./ (hbar); 
    totalcomp = -(term1real - term2real) ./ (hbar); 
    du(:,1:2*n_I ) = totalreal;
    du(:,2*n_I+1:4*n_I) = totalcomp;
    
    du = du(:);
end

function H = Hamiltonian(Bt)
    global hbar gamma_e gamma_n sigma_z sigma_y sigma_x tau_z tau_y tau_x H_int n_I;

    % Hamiltonian with the external B field
    He2 = -gamma_e * hbar / 2 * (sigma_x * Bt(1) + sigma_y * Bt(2) + sigma_z * Bt(3));
    Hn4 = -gamma_n * hbar / 2 * (tau_x * Bt(1) + tau_y * Bt(2) + tau_z * Bt(3));
    H_e = kron(eye(n_I), He2);
    H_n = kron(Hn4, eye(2));
    H = H_e + H_n + H_int;
end

function U = instantaneousEigStates(Iw, t)
    [U,~] = eig(Hamiltonian(FSField(Iw, t)));
    for i = 1:size(U,2)
        U(:,i) = U(:,i) ./ norm(U(:,i));
    end
end
