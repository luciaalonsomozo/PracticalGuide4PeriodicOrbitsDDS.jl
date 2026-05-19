############################################################################################################################################
# Rigorous computation of period-p orbits for the logistic map
# -----------------------------------------------------------
# This file proves the existence of period-p orbits in three steps:
# 1. We define the model and its derivatives;
#    we reformulate the problem as a zero-finding problem: F(x)=0.
# 2. We use Newton’s method to find a numerical zero of F: x_bar;
#    we find an approximate inverse of F(x_bar).
# 3. We compute the necessary bounds Y, Z_1 and Z_2 to apply the
#    Newton-Kantorovich Theorem and rigorously guarantee the
#    existence of true solutions nearby x_bar with error r>0.
# Extra Steps: Nontriviality of the orbit and stability
############################################################################################################################################


using RadiiPolynomial # Exports interval arithmetic, and `newton`


# Define the parameter μ and the period p we seek the periodic orbit
μ_str = "3.2"
p = 2


# ================================================================================================================================
# Step 1: Model definition and derivatives
# ================================================================================================================================

println("\n********************************** STEP 1: REFORMULATE THE PROBLEM ************************************************")

# Define the logistic map and its derivative
f(x, μ)  = μ * x * (exact(1) - x)
Df(x, μ) = μ * (exact(1) - exact(2)*x)

# Define the zero-finding map F(x, μ, p) for an orbit of period p
function F(x, μ, p)
    v = zeros(eltype(x), p)
    for i in 1:p
        v[i] = f(x[i], μ) - x[mod1(i+1, p)]  # It encodes the condition f(x[i]) = x[i+1]
    end
    return v
end

# Jacobian matrix DF of F(x, μ, p)
function DF(x, μ, p)
    M = zeros(eltype(x), p, p)
    for i in 1:p
        M[i,i] = Df(x[i], μ)
        M[i,mod1(i+1, p)] = exact(-1)
    end
    return M
end

# Define F and DF for Newton
F_DF(x) = (F(x, μ, p), DF(x, μ, p))

println("Succesfully defined the functions f, F and their derivatives.")


# ================================================================================================================================
# Step 2: Numerical approximation
# ================================================================================================================================

println("\n********************************** STEP 2:  NUMERICAL APPROXIMATION ***********************************************")

# Convert μ to an interval and take its midpoint
Iμ = parse(Interval{Float64}, μ_str)
μ = mid(Iμ)

# Apply Newton directly with a hard-coded initial guess
initial_guess = [0.51, 0.79]                               # Change this if Newton fails or other periods
x_bar, success = newton(F_DF, initial_guess, maxiter=50)

# Uncomment this to override Newton and follow the x_bar from the paper (up to 2 decimal places))
#x_bar = [0.51, 0.79]

println(success ? "Numerical solution: x_bar = $x_bar" :
                  "Newton failed to find a numerical periodic orbit.")

# Compute approximate inverse Jacobian
A = inv(DF(x_bar, μ, p))
println("Approximate inverse: A = ", A)


# ================================================================================================================================
# Step 3: Rigorous computation using interval arithmetic
# ================================================================================================================================

println("\n********************************** STEP 3: RIGOROUS - INTERVAL ARITHMETIC ******************************************")

# Encloses the approximate solution x_bar and approximate inverse A in an interval
Ix_bar = interval(x_bar)
IA = interval(A)

println("Interval of numerical solution: Ix_bar = ", Ix_bar,
        "\nInterval of approximate inverse: IA = ", IA)

# Define Y, Z_1 and Z_2 abstractly and rigorously compute such bounds
Y = norm(IA * F(Ix_bar, Iμ, p), 1)
Z₁ = opnorm(interval(I) - IA * DF(Ix_bar, Iμ, p), 1)
R = Inf
L = interval(2) * Iμ
Z₂ = L * opnorm(IA, 1)

println("\nThe bounds can be computed explicitly and they are given by:",
        "\nY  = ", Y,
        "\nZ1 = ", Z₁,
        "\nZ2 = ", Z₂)

# Check the existence of a radius r>0 in the Newton-Kantorovich theorem
r_m = (exact(1) - Z₁ - sqrt((1 - Z₁)^2 - exact(2) * Y * Z₂)) / Z₂
r_p = (exact(1) - Z₁ + sqrt((1 - Z₁)^2 - exact(2) * Y * Z₂)) / Z₂

println("\nThe admissible radii can be computed explicitly and they are given by:",
        "\nr_m = ", r_m,
        "\nr_p = ", r_p)

# Construct existence interval [r_m, r_p] and find an admissible radius r>0 for a true solution
r_interval = interval(sup(r_m), inf((1 - Z₁) / Z₂))
r = sup(r_m)

println("Admissible radii interval: ", r_interval,
        "\nExample of admissible radius r = ", r)


# ================================================================================================================================
# Extra Steps: Nontriviality and stability
# ================================================================================================================================

println("\n********************************** CHECKING PERIODICITY p = $p (RIGOROUS - INTERVAL ARITHMETIC) *********************")

# Construct validated enclosures around each coordinate
enclosures = [interval(Ix_bar[i], r; format=:midpoint) for i in eachindex(Ix_bar)]

# Check disjointness to rule out trivial solutions
if isdisjoint_interval(enclosures...)
    println("Period-$p orbit found in following disjoint intervals = ", enclosures)
else
    println("Period-$p orbit not found because the intervals are not disjoint.")
    return
end

println("\n********************************** CHECKING THE STABILITY (RIGOROUS - INTERVAL ARITHMETIC) *************************")

# The eigenvalue at the periodic orbit for n=1 is the product of derivatives along the orbit
function DG(x, μ)
    result = exact(1)
    for i in eachindex(x)
        result *= Df(x[i], μ)
    end
    return result
end

# Evaluate the eigenvalue using product of derivatives
λ = DG(enclosures, Iμ)

if isstrictless(interval(1), abs(λ))
    return println("The periodic orbit is UNSTABLE, as ellthe eigenvalue λ is contained in the following interval: ", λ)
elseif in_interval(1, abs(λ))
    return println("INCONCLUSIVE, as the absolute value of the eigenvalue |λ| lies in an interval containing the value 1: ", λ)
else
    return println("The periodic orbit is STABLE, as the eigenvalue λ is contained in the following interval: ", λ)
end

# ==============================================================================
# SCAN LOGIC FOR GENERATING THOUSANDS OF ORBITS (COMMENTED OUT)
# ==============================================================================
# function main()
#     periods = 2:80
#
#     # Parametric sweep over the interesting chaotic window of the Logistic Map
#     for mu_val in 3.4:0.0005:4.0
#         mu_str = string(round(mu_val, digits=4))
#
#         for n in periods
#             for attempt in 1:10   # Up to m verification attempts per (μ, n) configuration
#                 if run_main(mu_str, n) == true
#                     break        # Orbit verified! Move forward to next period size
#                 end
#             end
#         end
#     end
# end
# ==============================================================================
