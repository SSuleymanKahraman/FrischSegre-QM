using Plots;
using LinearAlgebra
using Statistics
using DifferentialEquations, ODEInterfaceDiffEq
using WignerD

# Electron spin inital state
field = 1;                      # 1 -> exact field, 2 -> quadrupole field 
ms_initial = 1;                 # 1 -> ms=-1/2, 2 -> ms=+1/2
# Display the selected parameters
println("ms_initial = " * string(ms_initial))
println("field = " * string(field))

# # PHYSICAL CONSTANTS from NIST 
hbar = 6.62607015e-34/2/pi ;    # Reduced Planck constant (J s)
mu_0 = 4*pi*1e-7 ;              # Vacuum permeability (Tm/A)
gamma_e = -1.76085963023e11 ;   # Electron gyromagnetic ratio  (1/sT). 

# 39K Constants
gamma_n = 1.25e7;               # Nuclear gyromagnetic ratio [gfactor*muB/hbar] (1/(Ts)) [http://triton.iqfr.csic.es/guide/eNMR/chem/]
spin_I = 3/2;                   # Nuclear spin
a_hfs = 230.8598601e6;          # Hz Arimondo1977 pg67

## FS experimental parameters
v = 800 ;                       # Atom speed (m/s)
za = 1.05e-4;                   # Wire position (m)
Br = 0.42 * 1e-4;               # Remnant field (T)
L_IR = 17.6e-3;           # Length of the inner rotation chamber (m)

# Experimental data in A vs probability
FS_data = [0.19, 6.14, 14.87, 26.68, 30.81, 26.8, 12.62, 0.1] / 100;
FS_Iwire = [0.010, 0.020, 0.03, 0.05, 0.10, 0.20, 0.30, 0.5];

# Wire currents to simulate
Iwire_list = FS_Iwire;

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
S_ex = kron(Matrix{Float64}(I, n_I, n_I), sigma_x);
S_nx = kron(tau_x, Matrix{Float64}(I, 2, 2));
S_ey = kron(Matrix{Float64}(I, n_I, n_I), sigma_y);
S_ny = kron(tau_y, Matrix{Float64}(I, 2, 2));
S_ez = kron(Matrix{Float64}(I, n_I, n_I), sigma_z);
S_nz = kron(tau_z, Matrix{Float64}(I, 2, 2));
sigma_int = (S_ex * S_nx + S_ey * S_ny + S_ez * S_nz) / 4;
A_hfs = a_hfs * hbar*2*pi;
H_int = A_hfs * sigma_int;

# Initial nuclear spin state mixture
rho_n_i = ones(n_I)/n_I;      # Maximally mixed
rho_i = zeros(2*n_I,2*n_I);
# Initial electron spin state
if ms_initial == 1
    rho_e_i = [1; 0];                   # Down spin, ms = -1/2
    # Combined initial state
    rho_i[1:n_I,1:n_I] = diagm(rho_n_i[end:-1:1]) .+ 0im;
else 
    rho_e_i = [0; 1];                   # Up spin, ms = +1/2
    # Combined initial state
    rho_i[n_I+1:2*n_I,n_I+1:2*n_I] = diagm(rho_n_i) .+ 0im;
end

# Flight time
tmax = +L_IR/v/2;  # final time
tmin = -L_IR/v/2;  # initial time
tspan = (tmin, tmax);

# Interval for saving each point during flight
time_save_step = 1e-8; # (unrelated to solver step size)
Nt = length(tmin:time_save_step:tmax);

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

function FSField(Iw, v, za, t, field)
    if field == 2
        # Null point
        t_NP = mu_0 * Iw / ( 2 * pi * Br * v);
        # Quadrupole field:
        Byt = za * Br^2 * 2 * pi / (mu_0 * Iw)
        Bzt = v * (t - t_NP) * Br^2 * 2 * pi / (mu_0 * Iw)
        Bxt = 0 
    else
        # Exact field:
        Byt = mu_0 * Iw / (2 * pi * (za^2 + (v * t)^2)) * za
        Bzt = Br - mu_0 *Iw / (2 * pi * (za^2 + (v * t)^2)) * v * t
        Bxt = 0 
    end
    return [Bxt;Byt;Bzt];
end


### Defnining the differential equation for the solver
function VonNeumann!(du, u, p, t)
    # p[1]: Iwire
    # p[2]: v
    # p[3]: za
    # u[:,1:2*n_I ]: real(kron(rhon,rhoe))
    # u[:,2*n_I+1:4*n_I]: imag(kron(rhon,rhoe))
    
    Bt = FSField(p[1],p[2],p[3],t,p[4]);

    H = Hamiltonian(Bt);

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



function Hamiltonian(Bt)
    # Hamiltonian with the external B field
    He2 = -gamma_e * hbar / 2 * (sigma_x * Bt[1] + sigma_y * Bt[2] + sigma_z * Bt[3]);
    Hn4 = -gamma_n * hbar / 2 * (tau_x * Bt[1] + tau_y * Bt[2] + tau_z * Bt[3]);
    H_e = kron(Matrix{ComplexF64}(I, n_I, n_I), He2);
    H_n = kron(Hn4, Matrix{ComplexF64}(I, 2, 2));
    H = H_e + H_n + H_int;
    return H;
end

function instantaneousEigStates(Iw, v, za, t, field)
    U = eigvecs(Hamiltonian(FSField(Iw, v, za, t, field)));
    for i = 1:size(U,2)
        U[:,i] = U[:,i] ./ norm(U[:,i]);
    end
    return U;
end


# Variable initialization
pe_flip = zeros(length(Iwire_list));
pe_flip_t = zeros(Nt,length(Iwire_list)) .+ 0.0im;
p_inst = zeros(2*n_I,Nt,length(Iwire_list));
rhos_inst = zeros(2*n_I,2*n_I,Nt,length(Iwire_list)) .+ 0.0im;

# Loop over currents
for iI = 1:lastindex(Iwire_list)
    # Print current iteration
    print("I_w = " * string(Iwire_list[iI]))

    # Instantaneous eigenstates at the entrance
    Ui = instantaneousEigStates(Iwire_list[iI], v, za, tmin, field);

    # Set initial density matrix
    rho0 = Ui * rho_i * Ui';
    u0 = zeros(2*n_I,4*n_I);
    u0[:,1:2*n_I] = real.(rho0);
    u0[:,2*n_I+1:4*n_I] = imag.(rho0);

    # Assign Iwire, v, za 
    params = [Iwire_list[iI], v, za, field];

    # Set up the ODE problem
    prob = ODEProblem(VonNeumann!, u0, tspan, params) 
    # Solve the problem
    @time rhot = solve(prob, alg, saveat=time_save_step, dtmin=dtmin, maxiters=maxiters, reltol=reltol, abstol=abstol, force_dtmin=force_dtmin, dt = 1e-10)
    
    # Calculate the relevant variables
    for it = 1:Nt
        # Complex density matrix
        rhototal = (rhot.u[it][:,1:2*n_I] .+ 1im * rhot.u[it][:,2*n_I+1:4*n_I]);
        
        # Density matrix in the instantaneous eigenstate basis
        Ut = instantaneousEigStates(Iwire_list[iI], v, za, rhot.t[it], field);
        rhos_inst[:,:,it,iI] = Ut' * rhototal * Ut;

        # Measurement probabilities of each state
        p_inst[:,it,iI] = real.(diag(rhos_inst[:,:,it,iI]));

        # Electron spin flip probability
        if ms_initial == 1
            pe_flip_t[it,iI] = sum(p_inst[(1+n_I):2*n_I, it, iI])
        else
            pe_flip_t[it,iI] = sum(p_inst[(1):n_I, it, iI])
        end

    end
    # Final flip probability 
    pe_flip[iI] = real(pe_flip_t[end,iI]);

    # Plot the evolution of the diagonal of the density matrix for a particular current
    if iI == 5
        p1 = Plots.plot();
        p1 = plot!(rhot.t*1e6, p_inst[:,:,iI]', xlabel="Time [us]", ylabel="Population", label=["Lowest energy state" "." "." "." "." "." "." "Highest energy state"], xlims=(tmin, tmax).*1e6, lw=:2, ls=:auto, legend =:left)
        display(p1);
    end

end

# # R-squared
SS_0 = sum((FS_data .- mean(FS_data)).^2);
SS_1 = sum((FS_data .- (pe_flip)).^2);
R2 = 1 .- SS_1./SS_0;
print("R^2 = " * string(round(1000*R2)/1000))

# # Analytical curve data
Iws = 10 .^(range(-2,stop=0,length=201));
# delta = za*abs(gamma_e)*( 2*pi./(mu_0*Iws)*Br^2 *za)/2/v;
Bx = 0; By = 0; Bz = 0; 
delta = abs(gamma_e)/(2*v) * (By*za .+ (By^2 + (Bz+Br)^2) * pi*za^2 ./ (mu_0*Iws) .+ (Bx^2+By^2)/(By^2+(Bz+Br)^2) * mu_0*Iws ./ (4*pi) );
yNP = v * mu_0 * Iwire_list * (Br) / ( 2 * pi * v * ( (Br )^2) );
# # Majorana solution
W_m = exp.(-2*pi*delta);
alpha = 2 * asin.(exp.(-pi*delta/4));
W_r = (rho_n_i[4]*WignerD.wignerdjmn.(2, -2, -1, alpha).^2 .+ rho_n_i[3]*WignerD.wignerdjmn.(2, -2, 0, alpha).^2 .+ rho_n_i[2]*WignerD.wignerdjmn.(2, -2, 1, alpha).^2 .+ rho_n_i[1]*WignerD.wignerdjmn.(2, -2, 2, alpha).^2);

# Plot the curve
p2 = Plots.plot();
p2 = scatter(Iwire_list, FS_data, mc=:black, shape=:circle, ms=:6, label = "FS data", xlabel="Wire current [A]", ylabel="Flip probability", title="R^2="*string(round(1000*R2)/1000)*" Br="*string(round(Br/1e-6))*"uT v="*string(round(v))*"m/s za="*string(round(za/1e-6))*"um", xaxis=:log) 
p2 = scatter!(Iwire_list, pe_flip, mc=:red, shape=:xcross, ms=:6, msw=:3, label = "Simulation", legend =:topleft) 
p2 = plot!(Iws, W_m, lc=:gray, linestyle=:dash, label = "Majorana");
p2 = plot!(Iws, W_r, lc=:blue, label = "Rabi");
display(p2);


