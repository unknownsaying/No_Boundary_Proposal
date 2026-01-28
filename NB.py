import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import fsolve
from scipy.special import gamma, factorial
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

plt.style.use('seaborn-v0_8-darkgrid')
np.set_printoptions(precision=4, suppress=True)

class NoBoundaryUniverse:
    """
    Implementation of Hartle-Hawking No-Boundary Proposal concepts.
    
    This class provides tools to visualize and compute key aspects of the 
    no-boundary wave function of the universe.
    """
    
    def __init__(self, Lambda=1.0, G=1.0, hbar=1.0):
        """
        Initialize with fundamental constants.
        
        Parameters:
        -----------
        Lambda : float
            Cosmological constant (in appropriate units)
        G : float
            Newton's constant
        hbar : float
            Reduced Planck constant
        """
        self.Lambda = Lambda
        self.G = G
        self.hbar = hbar
        
        # Derived parameters
        self.H = np.sqrt(self.Lambda/3)  # Hubble parameter
        
    def euclidean_friedmann(self, tau, a, model='de_sitter'):
        """
        Euclidean Friedmann equation for different cosmological models.
        
        Parameters:
        -----------
        tau : float or array
            Euclidean time
        a : float or array
            Scale factor
        model : str
            Cosmological model: 'de_sitter', 'closed', 'open', 'flat'
            
        Returns:
        --------
        da_dtau : float or array
            Derivative of scale factor with respect to Euclidean time
        d2a_dtau2 : float or array
            Second derivative
        """
        if model == 'de_sitter':
            # Pure cosmological constant: d²a/dτ² = H²a
            d2a_dtau2 = self.H**2 * a
            # First Friedmann in Euclidean signature: (da/dτ)² = 1 - H²a²
            da_dtau = np.sqrt(np.clip(1 - self.H**2 * a**2, 0, None))
            return da_dtau, d2a_dtau2
            
        elif model == 'closed':
            # Closed universe with matter
            rho = 1.0 / a**3  # Matter density ∝ a^-3
            d2a_dtau2 = (4*np.pi*self.G/3) * rho * a - self.H**2 * a
            da_dtau = np.sqrt(np.clip(1 - (8*np.pi*self.G/3) * rho * a**2 - self.H**2 * a**2, 0, None))
            return da_dtau, d2a_dtau2
            
        elif model == 'flat':
            # Flat universe (k=0)
            d2a_dtau2 = self.H**2 * a
            da_dtau = self.H * a
            return da_dtau, d2a_dtau2
            
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def solve_euclidean_trajectory(self, a_final, n_points=1000, model='de_sitter'):
        """
        Solve for the Euclidean scale factor trajectory from South Pole to final a.
        
        Parameters:
        -----------
        a_final : float
            Final scale factor (boundary condition)
        n_points : int
            Number of points in the trajectory
        model : str
            Cosmological model
            
        Returns:
        --------
        tau : array
            Euclidean time parameter
        a_tau : array
            Scale factor as function of Euclidean time
        """
        # Initial conditions at South Pole (τ=0)
        # a(0) = 0, a'(0) = 1 (smoothness/regularity condition)
        a0 = 1e-6  # Small but non-zero to avoid division by zero
        da0_dtau = 1.0
        
        # For de Sitter case, we can solve analytically
        if model == 'de_sitter':
            # Analytical solution: a(τ) = (1/H) * sin(Hτ)
            # Find τ such that a(τ) = a_final
            if a_final <= 1/self.H:
                tau_max = np.arcsin(self.H * a_final) / self.H
                tau = np.linspace(0, tau_max, n_points)
                a_tau = np.sin(self.H * tau) / self.H
                return tau, a_tau
            else:
                # Need complex contour for a_final > 1/H
                print(f"Warning: a_final={a_final:.4f} > 1/H={1/self.H:.4f}, using complex contour")
                return self.complex_contour_solution(a_final, n_points)
        
        # For other models, solve numerically
        def equations(y, t):
            a, da = y
            da_dtau, d2a_dtau2 = self.euclidean_friedmann(t, a, model)
            return [da, d2a_dtau2]
        
        # Find the Euclidean time to reach a_final
        def target(tau_max):
            sol = solve_ivp(equations, [0, tau_max], [a0, da0_dtau], 
                           t_eval=[0, tau_max], method='RK45')
            return sol.y[0, -1] - a_final
        
        # Use root finding to determine required τ_max
        tau_max_guess = np.pi/(2*self.H)  # Reasonable guess for de Sitter-like
        try:
            tau_max = fsolve(target, tau_max_guess)[0]
        except:
            tau_max = tau_max_guess
            
        # Solve full trajectory
        tau = np.linspace(0, tau_max, n_points)
        sol = solve_ivp(equations, [0, tau_max], [a0, da0_dtau], 
                       t_eval=tau, method='RK45')
        
        return tau, sol.y[0]
    
    def complex_contour_solution(self, a_final, n_points=1000):
        """
        Implement a complex time contour for a_final > 1/H.
        
        This represents the analytic continuation from Euclidean to Lorentzian signature.
        """
        # The contour: purely Euclidean up to τ = π/(2H), then Lorentzian
        tau_E_max = np.pi/(2*self.H)  # Max Euclidean evolution
        t_L_max = np.arccosh(self.H * a_final) / self.H  # Lorentzian time needed
        
        # Euclidean part
        n_E = int(n_points * 0.4)
        tau_E = np.linspace(0, tau_E_max, n_E)
        a_E = np.sin(self.H * tau_E) / self.H
        
        # Lorentzian part (after analytic continuation)
        n_L = n_points - n_E
        t_L = np.linspace(0, t_L_max, n_L)
        a_L = np.cosh(self.H * t_L) / self.H
        
        # Combine: Note the time parameter becomes complex
        # We'll represent this as two separate real arrays
        return (tau_E, a_E), (t_L, a_L)
    
    def compute_euclidean_action(self, tau, a_tau, model='de_sitter'):
        """
        Compute the Euclidean Einstein-Hilbert action for a given trajectory.
        
        S_E = (1/16πG) ∫ d^4x √g (R - 2Λ)
        In minisuperspace: S_E ∝ ∫ dτ [ -a (da/dτ)² - a + Λa³/3 ]
        """
        # Compute derivative
        da_dtau = np.gradient(a_tau, tau)
        
        if model == 'de_sitter':
            # Simplified minisuperspace action for de Sitter
            # Prefactor: 3π/(4G) for closed universe
            prefactor = 3*np.pi/(4*self.G)
            
            # Integrand
            integrand = -a_tau * da_dtau**2 - a_tau + (self.Lambda/3) * a_tau**3
            
            # Numerically integrate
            S_E = prefactor * np.trapz(integrand, tau)
            
            return S_E
        
        else:
            # More general calculation
            # Ricci scalar for FLRW: R = 6[(a''/a) + (a'/a)² + k/a²]
            # In Euclidean signature with k=1 (closed)
            a_prime = da_dtau
            a_double_prime = np.gradient(a_prime, tau)
            
            R = 6 * (a_double_prime/a_tau + (a_prime/a_tau)**2 + 1/a_tau**2)
            
            # Volume element: √g = a³ * sin²χ sinθ dτ dχ dθ dφ
            # After angular integration: volume ∝ a³
            integrand = a_tau**3 * (R - 2*self.Lambda)
            
            S_E = (1/(16*np.pi*self.G)) * np.trapz(integrand, tau)
            
            return S_E
    
    def wave_function_amplitude(self, a_final, model='de_sitter'):
        """
        Compute the no-boundary wave function amplitude Ψ(a) ≈ exp(-S_E/ħ).
        
        Parameters:
        -----------
        a_final : float or array
            Final scale factor(s)
        model : str
            Cosmological model
            
        Returns:
        --------
        Psi : complex or array
            Wave function amplitude(s)
        S_E : float or array
            Corresponding Euclidean action(s)
        """
        if np.isscalar(a_final):
            a_final = np.array([a_final])
            scalar_input = True
        else:
            scalar_input = False
            
        Psi = np.zeros(len(a_final), dtype=complex)
        S_E_vals = np.zeros(len(a_final))
        
        for i, a_f in enumerate(a_final):
            if a_f <= 1/self.H or model != 'de_sitter':
                # Purely Euclidean regime
                tau, a_tau = self.solve_euclidean_trajectory(a_f, model=model)
                S_E = self.compute_euclidean_action(tau, a_tau, model)
                Psi[i] = np.exp(-S_E/self.hbar)
                S_E_vals[i] = S_E
            else:
                # Mixed Euclidean-Lorentzian regime (requires complex contour)
                # For de Sitter: Ψ(a) ∝ exp(iS_L/ħ) where S_L is Lorentzian action
                # S_L = (1/H)[(H²a² - 1)^{3/2} - i] for a > 1/H
                S_L_real = (1/self.H) * ((self.H**2 * a_f**2 - 1)**1.5)
                S_L_imag = -1/self.H  # Constant imaginary part
                Psi[i] = np.exp(1j * S_L_real/self.hbar) * np.exp(-S_L_imag/self.hbar)
                S_E_vals[i] = S_L_imag + 1j*S_L_real  # Complex "action"
                
        if scalar_input:
            return Psi[0], S_E_vals[0]
        else:
            return Psi, S_E_vals
    
    def visualize_spacetime_manifold(self):
        """
        Visualize the no-boundary spacetime manifold as a 4-sphere (Euclidean de Sitter).
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 4D Spacetime manifold (projected to 3D)
        ax1 = fig.add_subplot(231, projection='3d')
        self._plot_4sphere(ax1)
        ax1.set_title("No-Boundary Spacetime: 4-Sphere $S^4$")
        ax1.set_xlabel("x₁")
        ax1.set_ylabel("x₂")
        ax1.set_zlabel("x₃")
        
        # 2. Scale factor evolution
        ax2 = fig.add_subplot(232)
        self._plot_scale_factor_evolution(ax2)
        ax2.set_title("Scale Factor Evolution")
        ax2.set_xlabel("Euclidean Time τ")
        ax2.set_ylabel("Scale Factor a(τ)")
        ax2.legend()
        
        # 3. Wave function probability
        ax3 = fig.add_subplot(233)
        self._plot_wave_function(ax3)
        ax3.set_title("No-Boundary Wave Function $|\Psi(a)|^2$")
        ax3.set_xlabel("Scale Factor a")
        ax3.set_ylabel("Probability $|\Psi|^2$")
        ax3.legend()
        
        # 4. Analytic continuation contour
        ax4 = fig.add_subplot(234)
        self._plot_complex_contour(ax4)
        ax4.set_title("Complex Time Contour")
        ax4.set_xlabel("Re(Time)")
        ax4.set_ylabel("Im(Time)")
        
        # 5. Euclidean vs Lorentzian regimes
        ax5 = fig.add_subplot(235)
        self._plot_regimes(ax5)
        ax5.set_title("Euclidean vs Lorentzian Regimes")
        ax5.set_xlabel("Scale Factor a")
        ax5.set_ylabel("Action S")
        
        # 6. Conceptual diagram
        ax6 = fig.add_subplot(236)
        self._plot_conceptual_diagram(ax6)
        ax6.set_title("No-Boundary Conceptual Diagram")
        ax6.set_xlabel("")
        ax6.set_ylabel("")
        
        plt.tight_layout()
        return fig
    
    def _plot_4sphere(self, ax):
        """Plot a 3D projection of a 4-sphere."""
        # Create a sphere in 3D (representing 4D sphere projection)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        
        # 3D sphere coordinates
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot the sphere
        ax.plot_surface(x, y, z, alpha=0.3, cmap=cm.coolwarm)
        
        # Mark South Pole (τ=0)
        ax.scatter([0], [0], [-1], color='red', s=100, label='South Pole (τ=0)')
        
        # Mark a latitude (3-sphere slice)
        theta = np.pi/4  # 45° latitude
        circle_x = np.sin(theta) * np.cos(u)
        circle_y = np.sin(theta) * np.sin(u)
        circle_z = np.cos(theta) * np.ones_like(u)
        ax.plot(circle_x, circle_y, circle_z, 'g-', linewidth=3, label='3-Sphere Slice')
        
        ax.legend()
        
    def _plot_scale_factor_evolution(self, ax):
        """Plot scale factor evolution in Euclidean and Lorentzian time."""
        # Euclidean part
        tau_E = np.linspace(0, np.pi/(2*self.H), 100)
        a_E = np.sin(self.H * tau_E) / self.H
        
        # Lorentzian part (after analytic continuation)
        t_L = np.linspace(0, 2/self.H, 100)
        a_L = np.cosh(self.H * t_L) / self.H
        
        ax.plot(tau_E, a_E, 'b-', linewidth=2, label='Euclidean: $a(τ) = H^{-1}\sin(Hτ)$')
        ax.plot(t_L + np.pi/(2*self.H), a_L, 'r--', linewidth=2, 
                label='Lorentzian: $a(t) = H^{-1}\cosh(Ht)$')
        
        # Mark transition point
        transition_tau = np.pi/(2*self.H)
        transition_a = 1/self.H
        ax.axvline(x=transition_tau, color='k', linestyle=':', alpha=0.5)
        ax.axhline(y=transition_a, color='k', linestyle=':', alpha=0.5)
        ax.scatter([transition_tau], [transition_a], color='purple', s=100, 
                  label='Nucleation: τ = π/(2H)')
        
    def _plot_wave_function(self, ax):
        """Plot the wave function amplitude."""
        a_vals = np.linspace(0.01, 3/self.H, 500)
        Psi, S_vals = self.wave_function_amplitude(a_vals)
        
        # Probability density
        prob_density = np.abs(Psi)**2
        
        ax.plot(a_vals, prob_density, 'b-', linewidth=2, label='$|\Psi(a)|^2$')
        
        # Mark Euclidean/Lorentzian transition
        transition_a = 1/self.H
        ax.axvline(x=transition_a, color='r', linestyle='--', alpha=0.7,
                  label=f'Transition: a = $H^{{-1}}$ = {1/self.H:.2f}')
        
        ax.set_yscale('log')
        
    def _plot_complex_contour(self, ax):
        """Plot the complex time contour for analytic continuation."""
        # The contour: τ from 0 to π/(2H) (purely imaginary/Euclidean)
        # then t from 0 onward (purely real/Lorentzian)
        
        # Euclidean segment
        tau_E = np.linspace(0, np.pi/(2*self.H), 100)
        im_time_E = tau_E  # Purely imaginary in Lorentzian view
        
        # Lorentzian segment
        t_L = np.linspace(0, 2/self.H, 100)
        re_time_L = t_L + np.pi/(2*self.H)
        
        # Plot contour
        ax.plot(np.zeros_like(tau_E), im_time_E, 'b-', linewidth=3, 
                label='Euclidean: τ pure imaginary')
        ax.plot(re_time_L, np.zeros_like(t_L), 'r--', linewidth=3,
                label='Lorentzian: t real')
        
        # Mark transition
        ax.scatter([0], [np.pi/(2*self.H)], color='purple', s=100, 
                  label='Wick rotation: τ → it')
        
        ax.set_xlim(-0.5, 3/self.H)
        ax.set_ylim(-0.1, 3/self.H)
        
    def _plot_regimes(self, ax):
        """Plot the action in Euclidean and Lorentzian regimes."""
        a_vals = np.linspace(0.1, 2/self.H, 200)
        
        # Euclidean action (a < 1/H)
        a_E = a_vals[a_vals <= 1/self.H]
        S_E = np.zeros_like(a_E)
        for i, a in enumerate(a_E):
            tau, a_tau = self.solve_euclidean_trajectory(a)
            S_E[i] = self.compute_euclidean_action(tau, a_tau)
        
        # Lorentzian action (a > 1/H) - real part
        a_L = a_vals[a_vals > 1/self.H]
        S_L_real = (1/self.H) * ((self.H**2 * a_L**2 - 1)**1.5)
        
        ax.plot(a_E, S_E, 'b-', linewidth=2, label='Euclidean Action $S_E$ (real)')
        ax.plot(a_L, S_L_real, 'r--', linewidth=2, 
                label='Lorentzian Action $S_L$ (real part)')
        
        ax.axvline(x=1/self.H, color='k', linestyle=':', alpha=0.5)
        
    def _plot_conceptual_diagram(self, ax):
        """Create a conceptual diagram of the no-boundary idea."""
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Draw Earth-like sphere
        earth_circle = Circle((0.5, 0.5), 0.4, fill=False, 
                             edgecolor='blue', linewidth=2)
        ax.add_patch(earth_circle)
        
        # Mark South Pole
        ax.scatter([0.5], [0.1], color='red', s=200, zorder=5)
        ax.text(0.5, 0.08, 'South Pole\n(a=0, τ=0)', 
               ha='center', va='top', fontsize=10)
        
        # Draw latitude lines (3-sphere slices)
        for lat in [0.3, 0.5, 0.7]:
            circle = Circle((0.5, 0.5), lat*0.4, fill=False, 
                           edgecolor='green', linestyle='--', alpha=0.7)
            ax.add_patch(circle)
            
        # Label the largest slice as our universe
        ax.text(0.9, 0.5, 'Our Universe\n(3-sphere slice)', 
               ha='left', va='center', fontsize=10)
        
        # Arrow showing "emergence of time"
        ax.arrow(0.5, 0.5, 0, -0.3, head_width=0.05, head_length=0.05, 
                fc='purple', ec='purple')
        ax.text(0.6, 0.35, 'Emergence of\ntime direction', 
               ha='left', va='center', fontsize=9)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
    def quantum_fluctuations(self, k_modes=10):
        """
        Calculate quantum fluctuations around the classical trajectory.
        
        These fluctuations become the seeds for cosmic structure formation.
        """
        # Wave numbers (comoving)
        k = np.arange(1, k_modes + 1)
        
        # Power spectrum for tensor fluctuations (gravitational waves)
        # In de Sitter: P_T(k) ∝ H²/M_pl²
        H = self.H
        M_pl = 1/np.sqrt(self.G)  # Planck mass
        
        P_T = (H**2)/(M_pl**2) * np.ones_like(k)
        
        # Scalar fluctuations (density perturbations)
        # P_S(k) ∝ (H⁴)/(φ̇²) where φ̇ is inflaton field time derivative
        # For pure de Sitter, we need an inflaton model
        phi_dot = H * M_pl  # Rough estimate
        P_S = (H**4)/(phi_dot**2) * np.ones_like(k)
        
        # Spectral indices (approximately scale-invariant for de Sitter)
        n_T = -2 * (self.Lambda/(3*H**2))  # Tensor tilt
        n_S = 1 - 2 * (self.Lambda/(3*H**2))  # Scalar tilt
        
        return {
            'k_modes': k,
            'tensor_power': P_T,
            'scalar_power': P_S,
            'tensor_tilt': n_T,
            'scalar_tilt': n_S,
            'tensor_to_scalar_ratio': P_T[0]/P_S[0]
        }
    
    def create_animation(self):
        """
        Create an animation showing the nucleation and expansion of the universe.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Setup for animation
        n_frames = 100
        tau_max = np.pi/(2*self.H)
        t_max = 2/self.H
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Fraction of evolution
            frac = frame / n_frames
            
            # Euclidean evolution (first half)
            if frac <= 0.5:
                tau_current = 2 * frac * tau_max
                tau_vals = np.linspace(0, tau_current, 100)
                a_vals = np.sin(self.H * tau_vals) / self.H
                
                ax1.plot(tau_vals, a_vals, 'b-', linewidth=3)
                ax1.set_xlim(0, tau_max)
                ax1.set_ylim(0, 1.2/self.H)
                ax1.set_xlabel("Euclidean Time τ")
                ax1.set_ylabel("Scale Factor a(τ)")
                ax1.set_title(f"Euclidean Regime: τ = {tau_current:.3f}")
                
                # Draw corresponding sphere
                theta = self.H * tau_current
                u = np.linspace(0, 2*np.pi, 50)
                circle_x = np.sin(theta) * np.cos(u)
                circle_y = np.sin(theta) * np.sin(u)
                
                ax2.plot(circle_x, circle_y, 'g-', linewidth=2)
                ax2.set_xlim(-1.2, 1.2)
                ax2.set_ylim(-1.2, 1.2)
                ax2.set_aspect('equal')
                ax2.set_title("4-Sphere Cross-section")
                
            else:
                # Lorentzian evolution (second half)
                t_current = 2 * (frac - 0.5) * t_max
                t_vals = np.linspace(0, t_current, 100)
                a_vals = np.cosh(self.H * t_vals) / self.H
                
                ax1.plot(t_vals + tau_max, a_vals, 'r-', linewidth=3)
                ax1.set_xlim(0, tau_max + t_max)
                ax1.set_ylim(0, 2.5/self.H)
                ax1.set_xlabel("Time")
                ax1.set_ylabel("Scale Factor a")
                ax1.set_title(f"Lorentzian Regime: t = {t_current:.3f}")
                
                # Draw expanding circle
                r = np.cosh(self.H * t_current) / self.H
                circle = plt.Circle((0, 0), r, fill=False, edgecolor='r', linewidth=2)
                ax2.add_patch(circle)
                ax2.set_xlim(-2.5/self.H, 2.5/self.H)
                ax2.set_ylim(-2.5/self.H, 2.5/self.H)
                ax2.set_aspect('equal')
                ax2.set_title("Expanding Universe")
            
            return ax1, ax2
        
        anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
        plt.close(fig)
        return anim
    
    def compute_observables(self, a_obs=1.0):
        """
        Compute observable quantities predicted by the no-boundary proposal.
        
        Parameters:
        -----------
        a_obs : float
            Scale factor at observation time
            
        Returns:
        --------
        dict : Dictionary of observable quantities
        """
        # Current Hubble parameter
        H0 = self.H * a_obs  # Proper Hubble parameter
        
        # Age of the universe (approximate)
        if a_obs > 1/self.H:
            t_age = np.arccosh(self.H * a_obs) / self.H
        else:
            t_age = np.arcsin(self.H * a_obs) / self.H
            
        # Horizon size
        horizon_size = 1/H0
        
        # Curvature parameter
        Omega_k = -1/(self.H**2 * a_obs**2)  # Negative for closed universe
        
        # Temperature (assuming radiation domination initially)
        T_initial = M_pl = 1/np.sqrt(self.G)
        T_current = T_initial / a_obs
        
        # Quantum fluctuation parameters
        fluct = self.quantum_fluctuations()
        
        return {
            'Hubble_parameter': H0,
            'Age_of_universe': t_age,
            'Horizon_size': horizon_size,
            'Curvature_parameter': Omega_k,
            'Current_temperature': T_current,
            'Tensor_to_scalar_ratio': fluct['tensor_to_scalar_ratio'],
            'Scalar_spectral_index': fluct['scalar_tilt'],
            'Tensor_spectral_index': fluct['tensor_tilt'],
            'No_boundary_probability': np.exp(-2/self.H)  # Approximate
        }


def demonstrate_no_boundary():
    """Demonstrate the no-boundary proposal with comprehensive visualizations."""
    
    print("="*70)
    print("HARTLE-HAWKING NO-BOUNDARY PROPOSAL DEMONSTRATION")
    print("="*70)
    print("\nInitializing universe with cosmological constant Λ=1.0...")
    
    # Create a no-boundary universe
    universe = NoBoundaryUniverse(Lambda=1.0, G=0.1, hbar=1.0)
    
    print(f"\n1. BASIC PARAMETERS:")
    print(f"   Hubble parameter H = √(Λ/3) = {universe.H:.4f}")
    print(f"   Hubble radius 1/H = {1/universe.H:.4f}")
    print(f"   Planck mass M_pl = 1/√G = {1/np.sqrt(universe.G):.4f}")
    
    print(f"\n2. SOLVING EUCLIDEAN TRAJECTORY:")
    a_test = 0.5/universe.H
    tau, a_tau = universe.solve_euclidean_trajectory(a_test)
    S_E = universe.compute_euclidean_action(tau, a_tau)
    print(f"   For a_final = {a_test:.4f} (Euclidean regime):")
    print(f"   Euclidean action S_E = {S_E:.4f}")
    print(f"   Wave function amplitude |Ψ| = {np.abs(np.exp(-S_E)):.4e}")
    
    print(f"\n3. WAVE FUNCTION CALCULATION:")
    a_values = np.array([0.3, 0.7, 1.0, 1.5, 2.0]) / universe.H
    Psi_vals, S_vals = universe.wave_function_amplitude(a_values)
    
    for a, psi, s in zip(a_values, Psi_vals, S_vals):
        regime = "Euclidean" if a <= 1/universe.H else "Lorentzian"
        print(f"   a = {a:.3f}/H: {regime} regime")
        print(f"     Action: {s:.4f}")
        print(f"     Ψ(a): {psi:.4e}")
        print(f"     |Ψ|²: {np.abs(psi)**2:.4e}")
    
    print(f"\n4. QUANTUM FLUCTUATIONS:")
    fluct = universe.quantum_fluctuations(k_modes=5)
    print(f"   Tensor-to-scalar ratio: r = {fluct['tensor_to_scalar_ratio']:.4e}")
    print(f"   Scalar spectral index: n_s = {fluct['scalar_tilt']:.4f}")
    print(f"   Tensor spectral index: n_t = {fluct['tensor_tilt']:.4f}")
    
    print(f"\n5. OBSERVABLE PREDICTIONS (for a_obs = 1/H):")
    obs = universe.compute_observables(a_obs=1.0/universe.H)
    for key, val in obs.items():
        print(f"   {key}: {val:.4e}")
    
    print(f"\n6. GENERATING VISUALIZATIONS...")
    
    # Create visualizations
    fig = universe.visualize_spacetime_manifold()
    plt.savefig('no_boundary_visualization.png', dpi=150, bbox_inches='tight')
    
    # Additional specialized plots
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Action as function of a
    ax1 = axes[0, 0]
    a_range = np.linspace(0.01, 2.5/universe.H, 300)
    Psi_range, S_range = universe.wave_function_amplitude(a_range)
    
    ax1.plot(a_range * universe.H, np.real(S_range), 'b-', label='Re(S)')
    ax1.plot(a_range * universe.H, np.imag(S_range), 'r--', label='Im(S)')
    ax1.axvline(x=1, color='k', linestyle=':', alpha=0.5, label='a = 1/H')
    ax1.set_xlabel('Scale Factor (units of 1/H)')
    ax1.set_ylabel('Action S')
    ax1.set_title('Complex Action vs Scale Factor')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Probability distribution
    ax2 = axes[0, 1]
    prob = np.abs(Psi_range)**2
    ax2.semilogy(a_range * universe.H, prob, 'g-', linewidth=2)
    ax2.axvline(x=1, color='k', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Scale Factor (units of 1/H)')
    ax2.set_ylabel('Probability |Ψ(a)|²')
    ax2.set_title('No-Boundary Probability Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Phase of wave function
    ax3 = axes[1, 0]
    phase = np.angle(Psi_range)
    ax3.plot(a_range * universe.H, phase, 'm-', linewidth=2)
    ax3.axvline(x=1, color='k', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Scale Factor (units of 1/H)')
    ax3.set_ylabel('Phase of Ψ(a) (radians)')
    ax3.set_title('Quantum Phase of the Universe')
    ax3.grid(True, alpha=0.3)
    
    # Power spectrum
    ax4 = axes[1, 1]
    k_modes = np.logspace(-1, 1, 100)
    ax4.loglog(k_modes, fluct['scalar_power'][0] * (k_modes/k_modes[0])**(fluct['scalar_tilt']-1), 
               'b-', label='Scalar P(k)')
    ax4.loglog(k_modes, fluct['tensor_power'][0] * (k_modes/k_modes[0])**(fluct['tensor_tilt']), 
               'r--', label='Tensor P(k)')
    ax4.set_xlabel('Wave number k')
    ax4.set_ylabel('Power Spectrum P(k)')
    ax4.set_title('Primordial Power Spectra')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('no_boundary_quantities.png', dpi=150, bbox_inches='tight')
    
    print(f"\n7. CONCEPTUAL SUMMARY:")
    print(f"   • The universe emerges from a compact Euclidean 4-geometry")
    print(f"   • Time emerges via analytic continuation: τ → it")
    print(f"   • The 'Big Bang' is replaced by a smooth 'South Pole'")
    print(f"   • Quantum probability peaks at small a, then becomes oscillatory")
    print(f"   • Predicts scale-invariant perturbations with r ≈ {fluct['tensor_to_scalar_ratio']:.2e}")
    
    print(f"\nVisualizations saved as 'no_boundary_visualization.png' and 'no_boundary_quantities.png'")
    print("="*70)
    
    return universe


# Run the demonstration
if __name__ == "__main__":
    universe = demonstrate_no_boundary()
    
    # Show one of the figures
    plt.figure(figsize=(10, 8))
    universe._plot_scale_factor_evolution(plt.gca())
    plt.title("No-Boundary Scale Factor Evolution")
    plt.show()
