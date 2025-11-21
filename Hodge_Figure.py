import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# --- Core Function: Spectral Coherence Coefficient C_N ---
def compute_C_stats(s, Ns):
    """
    Computes mean and variance of C_N for a sequence s over various window sizes N.
    s: array of normalized gaps (stationary, mean ~ 1)
    Ns: list of window sizes
    """
    stats = []
    for N in Ns:
        # Calculate C_N for all valid windows
        # Sliding window implementation
        # We want C_N = (sum_{k=0}^{N-2} s_{i+k}) / (sum_{k=0}^{N-1} s_{i+k})
        # This matches the definition: sum of first N-1 / sum of N
        
        # Efficient convolution-like approach
        # However, for clarity and correctness with variable N, simple loop with stride is robust
        
        c_values = []
        # Use a stride to reduce correlation between windows, e.g., N
        stride = max(1, N) 
        for i in range(0, len(s) - N, stride):
            window = s[i : i+N]
            num = np.sum(window[:-1])
            den = np.sum(window)
            if den > 0:
                c_values.append(num / den)
        
        c_values = np.array(c_values)
        if len(c_values) > 0:
            mean_c = np.mean(c_values)
            var_c = np.var(c_values)
            stats.append({'N': N, 'mean': mean_c, 'var': var_c, 'count': len(c_values)})
    
    return pd.DataFrame(stats)

# --- Simulation of Spectra ---

# 1. Algebraic Class (Geometric/Rigid)
# Modeled by GUE-like rigidity (short-range correlations). 
# We use an AR(1) process with negative correlation to mimic level repulsion.
def generate_algebraic_gaps(n_gaps):
    # AR(1): x_t = phi * x_{t-1} + noise
    # phi ~ -0.36 is characteristic of GUE/Riemann/YM stable gaps
    phi = -0.36
    noise = np.random.normal(1, 0.3, n_gaps) # Mean 1 to keep gaps positive roughly
    gaps = np.zeros(n_gaps)
    gaps[0] = 1.0
    for t in range(1, n_gaps):
        gaps[t] = 1.0 + phi * (gaps[t-1] - 1.0) + (noise[t] - 1.0)
    
    # Ensure positivity (simple clipping for this heuristic model)
    gaps = np.maximum(gaps, 0.01)
    # Normalize to exact mean 1
    gaps = gaps / np.mean(gaps)
    return gaps

# 2. Transcendental Class (Non-Algebraic/Chaotic)
# Modeled by a process with weaker or no level repulsion (e.g., Poisson or long-range noise).
# Here we mix some "colored noise" to simulate long-range correlations (1/f noise).
def generate_transcendental_gaps(n_gaps):
    # Generate 1/f^alpha noise (pink noise-like)
    # This creates long-range correlations
    white = np.random.normal(0, 1, n_gaps)
    freqs = np.fft.rfftfreq(n_gaps)
    # spectral density S(f) ~ 1/f (alpha=1)
    # amplitude ~ 1/sqrt(f)
    with np.errstate(divide='ignore'):
        scale = 1.0 / np.sqrt(np.maximum(freqs, 1e-10))
    scale[0] = 0
    pink = np.fft.irfft(np.fft.rfft(white) * scale, n=n_gaps)
    
    # Transform to positive gaps
    gaps = np.exp(pink) 
    gaps = gaps / np.mean(gaps)
    return gaps

# --- Main Execution ---

# Parameters
n_gaps = 100000
window_sizes = [5, 10, 20, 50, 100, 200]

# Generate Data
s_alg = generate_algebraic_gaps(n_gaps)
s_trans = generate_transcendental_gaps(n_gaps)

# Compute Statistics
df_alg = compute_C_stats(s_alg, window_sizes)
df_trans = compute_C_stats(s_trans, window_sizes)

# Theoretical Predictions
df_alg['theory_mean'] = (df_alg['N'] - 1) / df_alg['N']
df_alg['theory_var_slope'] = df_alg['var'].iloc[0] * (df_alg['N'].iloc[0] / df_alg['N'])**2 # Scale from first point

# --- Plotting ---

# Figure H1: Mean C_N vs N (Algebraic Case)
plt.figure(figsize=(8, 5))
plt.plot(df_alg['N'], df_alg['mean'], 'o-', label='Simulated (Algebraic)', color='blue')
plt.plot(df_alg['N'], df_alg['theory_mean'], 'x--', label='Theory (N-1)/N', color='red')
plt.xlabel('Window Size N')
plt.ylabel('Mean Coherence $E[C_N]$')
plt.title('H1. Mean Spectral Coherence vs N (Algebraic/Rigid Spectrum)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('Fig_H1_Mean.png')

# Figure H2: Variance vs N (Log-Log) - Algebraic
plt.figure(figsize=(8, 5))
plt.loglog(df_alg['N'], df_alg['var'], 'o-', label='Simulated (Algebraic)', color='blue')
# Reference line for N^-2
ref_x = np.array(window_sizes)
ref_y = df_alg['var'].iloc[0] * (ref_x[0] / ref_x)**2
plt.loglog(ref_x, ref_y, 'k--', label='Reference $N^{-2}$')
plt.xlabel('Window Size N (log)')
plt.ylabel('Variance Var($C_N$) (log)')
plt.title('H2. Variance of Coherence (Algebraic Stability)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('Fig_H2_Variance.png')

# Figure H3: Stress Test (Algebraic vs Transcendental)
plt.figure(figsize=(8, 5))
plt.loglog(df_alg['N'], df_alg['var'], 'o-', label='Algebraic (Rigid)', color='blue')
plt.loglog(df_trans['N'], df_trans['var'], 's-', label='Transcendental (Disordered)', color='orange')
plt.xlabel('Window Size N (log)')
plt.ylabel('Variance Var($C_N$) (log)')
plt.title('H3. Bridge A: Signature of Transcendental Disorder')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('Fig_H3_Perturbation.png')

# Output the data table for the user
print(df_alg[['N', 'mean', 'var']])