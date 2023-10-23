using Plots;
using MAT
using LinearAlgebra
using Statistics
using FileIO
using Dates
using JLD2
using DifferentialEquations, ODEInterfaceDiffEq

gr();
cd(@__DIR__)
print(@__FILE__)
println()

# To save the results
savefiles = true;
if savefiles
    # save results to a file with unique datetime savefilename
    dirpath = string(@__DIR__);
    savefilename = dirpath * "\\Output_FS_QM_model_" * string(Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS"));
    mkdir(savefilename);
    # copy current code to a new file
    cp(@__FILE__, savefilename * "\\executedscript.jl");
end

##########################################################
# Parameters that might be scanned later
iB = 1;     # iB = 2 -> quadrupole, iB = 1 -> exact for FS
iA = 1;     # Interaction coefficient 1-8, 1 is the correct value, 8 is no interaction
irhon = 1;  # Nuclear density matrices, up to 28
##########################################################

# # PHYSICAL CONSTANTS from NIST
# RSU : Relative Standard Uncertainty
hbar = 6.62607015e-34/2/pi ;    # Reduced Planck constant (J s)
mu_0 = 4*pi*1e-7 ;              # Vacuum permeability (Tm/A)
gamma_e = -1.76085963023e11 ;   # Electron gyromagnetic ratio  (1/sT). Relative Standard Uncertainty = 3.0e-10
# mu_e = -9.2847677043e-24 ;    # Electron magnetic moment (J/T). RSU = 3.0e-10

# 39K Constants
gamma_n = 1.25e7;               # Nuclear gyromagnetic ratio [gfactor*muB/hbar] (1/(Ts)) [http://triton.iqfr.csic.es/guide/eNMR/chem/]
R = 275e-12;                    # van der Waals atomic radius (m) [https://periodic.lanl.gov/3.shtml]
# mu_n = 1.96e-27;              # Nuclear magnetic moment (J/T)  [https://www-nds.iaea.org/nuclearmoments/][https://www-nds.iaea.org/publications/indc/indc-nds-0794/]
spin_I = 3/2;                   # Nuclear spin
a_hfs = 230.8598601e6;          # Hz Arimondo1977 pg67

# Spin-1/2 Pauli matrices
sigma_x = [0 1; 1 0] .+ 0.0im;
sigma_y = [0 -1im; 1im 0] .+ 0.0im;
sigma_z = [1 0; 0 -1] .+ 0.0im;

# Nuclear Hilbert space dimension
n_I = trunc(Int, 2 * spin_I + 1);
# Nuclear Pauli matrices
sm = spin_I:-1:-spin_I;
tau_z = 2*diagm(sm);
tauplus = diagm(1 => sqrt.((spin_I .- sm[2:end]).*(spin_I .+ sm[2:end] .+ 1)));
tauminus = diagm(-1 => sqrt.((spin_I .+ sm[1:end-1]).*(spin_I .- sm[1:end-1] .+ 1)));
tau_x = 2*(tauplus+tauminus)/2;
tau_y = 2*(-1im*tauplus+1im*tauminus)/2;

# 3/2 - 1/2 interaction matrix
S_ex = kron(Matrix{Float64}(I, n_I, n_I), sigma_x)
S_nx = kron(tau_x, Matrix{Float64}(I, 2, 2))
S_ey = kron(Matrix{Float64}(I, n_I, n_I), sigma_y)
S_ny = kron(tau_y, Matrix{Float64}(I, 2, 2))
S_ez = kron(Matrix{Float64}(I, n_I, n_I), sigma_z)
S_nz = kron(tau_z, Matrix{Float64}(I, 2, 2))
sigma_int = (S_ex * S_nx + S_ey * S_ny + S_ez * S_nz) / 4

## FS experimental parameters
v = 800 ;                   # Atom speed (m/s)
za = 1.05e-4;               # Wire position (m)
Br = 0.42 * 1e-4;           # Remnant field (T)

# Experimental data in A vs probability
FS_data = [0.19, 6.14, 14.87, 26.68, 30.81, 26.8, 12.62, 0.1] / 100;
FS_Iwire = [0.010, 0.020, 0.03, 0.05, 0.10, 0.20, 0.30, 0.5];

# Wire currents to simulate
Iwire_list = FS_Iwire

# Converting operators to the tensor space
Sigma_z = kron(Matrix{ComplexF64}(I, n_I, n_I), sigma_z) / 2
Tau_z = kron(tau_z, Matrix{ComplexF64}(I, 2, 2)) / 2
Pe_up = kron(Matrix{ComplexF64}(I, n_I, n_I), [1 0; 0 0]) .+ 0.0im;

# Initial density matrix for electron
rho_e_list = zeros(2,2,2);
# Up state
rho_e_list[:,:,1] = [1 0; 0 0];
# Down state
rho_e_list[:,:,2] = [0 0; 0 1];

# Initial density matrices for nucleus
rho_n_list = zeros(n_I,n_I,28);
# 4 level, iso, maximally mixed
rho_n_list[:,:,1] = Matrix{Float64}(I, n_I, n_I)/4;
# 4 level, iso, pure
rho_n_list[:,:,2] = ones(n_I,n_I)/4;
# 2 level, iso, mixed
temp = zeros(n_I);
temp[[1, end]] .= 1;
temp = temp./ sum(temp);
rho_n_list[:,:,3] = diagm(temp);
# 2 level, iso, pure
temp = zeros(n_I);
temp[[1, end]] .= sqrt.(1);
temp = temp./ norm(temp);
rho_n_list[:,:,4] = temp * temp';
# 4 level, aniso, mixed
temp = (1 .+(-spin_I:1:spin_I)/spin_I).^2;
temp = temp./ sum(temp);
rho_n_list[:,:,5] = diagm(temp);
# 4 level, aniso, pure
temp = (1 .+(-spin_I:1:spin_I)/spin_I);
temp = temp ./ norm(temp);
rho_n_list[:,:,6] = temp * temp';
# 4 level, aniso, mixed
temp = (1 .-(-spin_I:1:spin_I)/spin_I).^2;
temp = temp./ sum(temp);
rho_n_list[:,:,7] = diagm(temp);
# 4 level, aniso, pure
temp = (1 .-(-spin_I:1:spin_I)/spin_I);
temp = temp ./ norm(temp);
rho_n_list[:,:,8] = temp * temp';
# 4 level, aniso, mixed
temp = (1 .+(-spin_I:1:spin_I)/spin_I);
temp = temp./ sum(temp);
rho_n_list[:,:,9] = diagm(temp);
# 4 level, aniso, pure
temp = sqrt.(1 .+(-spin_I:1:spin_I)/spin_I);
temp = temp ./ norm(temp);
rho_n_list[:,:,10] = temp * temp';
# 4 level, aniso, mixed
temp = (1 .-(-spin_I:1:spin_I)/spin_I);
temp = temp./ sum(temp);
rho_n_list[:,:,11] = diagm(temp);
# 4 level, aniso, pure
temp = sqrt.(1 .-(-spin_I:1:spin_I)/spin_I);
temp = temp ./ norm(temp);
rho_n_list[:,:,12] = temp * temp';
# 4 level, aniso, mixed
temp = (1:1:n_I)/(n_I*(n_I+1)/2);
temp = temp./ sum(temp);
rho_n_list[:,:,13] = diagm(temp);
# 4 level, aniso, pure
temp = sqrt.((1:1:n_I)/(n_I*(n_I+1)/2));
temp = temp ./ norm(temp);
rho_n_list[:,:,14] = temp * temp';
# 4 level, aniso, mixed
temp = (n_I:-1:1)/(n_I*(n_I+1)/2);
temp = temp./ sum(temp);
rho_n_list[:,:,15] = diagm(temp);
# 4 level, aniso, pure
temp = sqrt.((n_I:-1:1)/(n_I*(n_I+1)/2));
temp = temp ./ norm(temp);
rho_n_list[:,:,16] = temp * temp';
# 4 level, aniso, mixed
temp = (1 .+(-spin_I:1:spin_I)/sqrt(spin_I*(spin_I+1)));
temp = temp./ sum(temp);
rho_n_list[:,:,17] = diagm(temp);
# 4 level, aniso, pure
temp = (1 .+(-spin_I:1:spin_I)/sqrt(spin_I*(spin_I+1)));
temp = temp ./ norm(temp);
rho_n_list[:,:,18] = temp * temp';
# 4 level, aniso, mixed
temp = (1 .-(-spin_I:1:spin_I)/sqrt(spin_I*(spin_I+1)));
temp = temp./ sum(temp);
rho_n_list[:,:,19] = diagm(temp);
# 4 level, aniso, pure
temp = (1 .-(-spin_I:1:spin_I)/sqrt(spin_I*(spin_I+1)));
temp = temp ./ norm(temp);
rho_n_list[:,:,20] = temp * temp';
# 2 level, aniso, mixed
temp = zeros(n_I);
temp[[1, end]] = [1 3];
temp = temp./ sum(temp);
rho_n_list[:,:,21] = diagm(temp);
# 2 level, aniso, pure
temp = zeros(n_I);
temp[[1, end]] = sqrt.([1 3]);
temp = temp./ norm(temp);
rho_n_list[:,:,22] = temp * temp';
# 2 level, aniso, mixed
temp = zeros(n_I);
temp[[1, end]] = [3 1];
temp = temp./ sum(temp);
rho_n_list[:,:,23] = diagm(temp);
# 2 level, aniso, pure
temp = zeros(n_I);
temp[[1, end]] = sqrt.([3 1]);
temp = temp./ norm(temp);
rho_n_list[:,:,24] = temp * temp';
# 2 level, aniso, mixed
temp = zeros(n_I);
temp[[1, end]] = [1 2];
temp = temp./ sum(temp);
rho_n_list[:,:,25] = diagm(temp);
# 2 level, aniso, pure
temp = zeros(n_I);
temp[[1, end]] = sqrt.([1 2]);
temp = temp./ norm(temp);
rho_n_list[:,:,26] = temp * temp';
# 2 level, aniso, mixed
temp = zeros(n_I);
temp[[1, end]] = [2 1];
temp = temp./ sum(temp);
rho_n_list[:,:,27] = diagm(temp);
# 2 level, aniso, pure
temp = zeros(n_I);
temp[[1, end]] = sqrt.([2 1]);
temp = temp./ norm(temp);
rho_n_list[:,:,28] = temp * temp';


### Interaction coefficients
A_list = zeros(8);
# Experimental value at https://link.aps.org/doi/10.1103/RevModPhys.49.31
A_list[1] = a_hfs * hbar*2*pi;
# Top-hat, self-averaged
A_list[2] = 1/2 * (-mu_0*gamma_e*gamma_n/(pi*R^3) * hbar^2);
# Gaussian, self-averaged
A_list[3] = 16/(3*pi^2) * (-mu_0*gamma_e*gamma_n/(pi*R^3) * hbar^2);
# Hartree self-averaged
A_list[4] = 28.4/3 * (-mu_0*gamma_e*gamma_n/(pi*R^3) * hbar^2);
# Top-hat, torque-averaged
A_list[5] = 5/16 * (-mu_0*gamma_e*gamma_n/(pi*R^3) * hbar^2);
# Gaussian, torque-averaged
A_list[6] = 4/(3*pi^2) * (-mu_0*gamma_e*gamma_n/(pi*R^3) * hbar^2);
# Hartree, torque-averaged
A_list[7] = 0.138 * (-mu_0*gamma_e*gamma_n/(pi*R^3) * hbar^2);
# Majorana case
A_list[8] = 0;

# Chooses initial eletron state according to field
if iB == 1
    irhoe = 2;  # Start from down spin for exact field
else
    irhoe = 1;  # Start from up spin for quadrupole field
end
rhoe0 = rho_e_list[:,:,irhoe];
# Initial nuclear state
rhon0 = rho_n_list[:,:,irhon];
# Initial density matrix in tensor space
rho0 = kron(rhon0,rhoe0) .+ 0im;
# Hyperfine coefficient
A = A_list[iA];
H_int = A * sigma_int;

# Flight time
tmax = 11e-6;   # final time
tmin = -11e-6;  # initial time
tspan = (tmin, tmax);

# Display the selected parameters
print("irhon ")
display(irhon)
print("irhoe ")
display(irhoe)
print("iB ")
display(iB)
print("iA ")
display(iA)

# Interval for saving each point during flight
time_save_step = 1e-8; # (unrelated to solver step size)

### Choosing the algorithm for the solver
alg = radau5();
# alg = Rosenbrock23();
# alg = SSPRK22(); dt = 1e-10;
# alg = Rosenbrock23();
# alg = radau();
# alg = TRBDF2();
# alg = RadauIIA3();
# alg = ImplicitEuler();
# alg = SSPSDIRK2();
# alg = ROCK2();

# Solver params
reltol = 1e-8;   #1e-14  # relative tolerance
abstol = 1e-8;   #1e-25  # absolute tolerance
maxiters = 1e12;
dtmin = 1e-40;           # minimum time step
force_dtmin = false;


### Defnining the differential equation for the solver
function VonNeumann!(du, u, p, t)
    # p[1]: Iwire
    # p[2]: v
    # p[3]: za
    # u[:,1:2*n_I ]: real(kron(rhon,rhoe))
    # u[:,2*n_I+1:4*n_I]: imag(kron(rhon,rhoe))
    
    if iB == 2
        # Quadrupole field:
        Byt = p[3] * Br^2 * 2 * pi / (mu_0 * p[1])
        Bzt = p[2] * t * Br^2 * 2 * pi / (mu_0 * p[1])
        Bxt = 0
    else
        # Exact field:
        Byt = mu_0 * p[1] / (2 * pi * (p[3]^2 + (p[2] * t)^2)) * p[3] 
        Bzt = Br - mu_0 * p[1] / (2 * pi * (p[3]^2 + (p[2] * t)^2)) * p[2] * t
        Bxt = 0 
    end

    # Hamiltonian with the external B field
    He2 = -gamma_e * hbar / 2 * (sigma_x * Bxt + sigma_y * Byt + sigma_z * Bzt);
    Hn4 = -gamma_n * hbar / 2 * (tau_x * Bxt + tau_y * Byt + tau_z * Bzt);
    H_e = kron(Matrix{ComplexF64}(I, n_I, n_I), He2);
    H_n = kron(Hn4, Matrix{ComplexF64}(I, 2, 2));
    H = H_e + H_n + H_int;

    # von Neumann equation
    # du = (H*u .- u*H) ./ (1im*hbar);
    term1real = real.(H) * u[:,1:2*n_I ] - imag.(H) * u[:,2*n_I+1:4*n_I];
    term2real = u[:,1:2*n_I ] * real.(H) - u[:,2*n_I+1:4*n_I] * imag.(H);
    term1comp = imag.(H) * u[:,1:2*n_I ] + real.(H) * u[:,2*n_I+1:4*n_I];
    term2comp = u[:,2*n_I+1:4*n_I] * real.(H) + u[:,1:2*n_I ] * imag.(H);
    totalreal =  (term1comp - term2comp) ./ (hbar); 
    totalcomp = -(term1real - term2real) ./ (hbar); 
    du[:,1:2*n_I ] = totalreal;
    du[:,2*n_I+1:4*n_I] = totalcomp;

end

# Splitting real and imaginary parts since its not natively supoorted
u0 = zeros(2*n_I,4*n_I);
u0[:,1:2*n_I ] = real.(rho0);
u0[:,2*n_I+1:4*n_I] = imag.(rho0);

# Variable initialization
pe_ups = zeros(length(Iwire_list));

# Loop over currents
for iI = 1:lastindex(Iwire_list)
    # Print current iteration
    print("iI ")
    display(iI);

    # Assign Iwire, v, za 
    p_i = [Iwire_list[iI], v, za]; # (these normally don't change unless multiple sweeps are done)

    # Set up the ODE problem
    prob = ODEProblem(VonNeumann!, u0, tspan, p_i) 
    # Solve the problem
    @time rhot = solve(prob, alg, saveat=time_save_step, dtmin=dtmin, maxiters=maxiters, reltol=reltol, abstol=abstol, force_dtmin=force_dtmin, dt = 1e-10)
    
    # Number of saved time steps
    Nt0 = length(rhot.t) # (different from number of solver steps)

    # Variable initialization
    Tau_z_t = zeros(Nt0) .+ 0.0im;
    Sigma_z_t = zeros(Nt0) .+ 0.0im;
    pe_up_t = zeros(Nt0) .+ 0.0im;

    # Calculate the relevant variables
    for it = 1:Nt0
        # Complex density matrix
        rhototal = Matrix{ComplexF64}(I, n_I*2, n_I*2);
        rhototal = (rhot.u[it][:,1:2*n_I] .+ 1im * rhot.u[it][:,2*n_I+1:4*n_I]);
        
        # Spin measurements
        Tau_z_t[it] = tr(rhototal * Tau_z);
        Sigma_z_t[it] = tr(rhototal * Sigma_z);
        pe_up_t[it] = tr(rhototal * Pe_up);
    end
    # Average over the final part of the flight which converges at later times
    pe_ups[iI] = real(mean(pe_up_t[trunc(Int64, 7*Nt0/8):Nt0]));

    # Plot the spin trajectories for a particular current
    if iI == 5
        p1 = Plots.plot();
        p1 = plot(rhot.t, real.(Sigma_z_t), linecolor=:blue, xlabel="Time [s]", label="<S_x>", xlims=(tmin, tmax)) # , legend =:best, label = "θ_n"
        p1 = plot!(rhot.t, real.(Tau_z_t), linecolor=:red, xlabel="Time [s]", label="<T_x>", xlims=(tmin, tmax)) # , legend =:best, label = "θ_n"
        display(p1);
        if savefiles
            savefig(p1, savefilename * "\\timetrace_iA-" * string(iA) * "_irhon-" * string(irhon) * "_irhoe-" * string(irhoe) * "_iB-" * string(iB) * ".png");
        end
    end

end

# # R-squared
SS_0 = sum((FS_data .- mean(FS_data)).^2);
SS_1 = sum((FS_data .- pe_ups).^2);
R2 = 1 .- SS_1./SS_0;
print("R2*100 = ")
println(R2 * 100)

# # Pearson correlation
rpearson = cor(pe_ups, FS_data)

# # Mean squared error
mse = sum((pe_ups-FS_data).^2)

# Plot the curve
p2 = Plots.plot();
p2 = scatter(Iwire_list, FS_data, linecolor=:blue, label = "FS data", xlabel="I", ylabel="Flip probability", xaxis=:log, title="R^2="*string(R2)*" rhon"*string(irhon)*" A"*string(iA)) #, legend =:best, label = "θ_e"
p2 = scatter!(Iwire_list, pe_ups, linecolor=:red, label = "Simulation", xlabel="I", ylabel="Flip probability", legend =:topleft, xaxis=:log) # , legend =:best, label = "θ_n"
display(p2);

# Save figure
if savefiles
    savefig(p2, savefilename * "\\flipvscurrent_iA-" * string(iA) * "_irhon-" * string(irhon) * "_irhoe-" * string(irhoe) * "_iB-" * string(iB) * ".png");
    workspacefilename = savefilename * "\\workspace_iA-" * string(iA) * "_irhon-" * string(irhon) * "_irhoe-" * string(irhoe) * "_iB-" * string(iB) * ".jld";
    @save workspacefilename
end

