
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import BSpline
from shapely.geometry import Polygon


SPLINE_DEGREE = 2 

## 📐 I. Geometry and Spline Functions
#-------------------------------------------------------------------------------

class SplineGeometry:

    def __init__(self, n_cp, k=SPLINE_DEGREE, npt=1000):
        self.n_cp = n_cp
        self.k = k
        self.npt = npt
        self.knot_vector = self.create_clamped_knot_vector(n_cp, k)

    def create_clamped_knot_vector(self, n_cp, k=SPLINE_DEGREE):
        """
        Generate a clamped B-spline knot vector: [0, 0, ..., u_i, ..., 1, 1]
        The end knots (0 and 1) are repeated k times to ensure the spline passes 
        through the first and last control points.
        Total length = n_cp + k + 1
        """
        n_knots = n_cp + k + 1
        # Number of internal knots required for the total length
        n_internal_knots = n_knots - 2 * k
        
        # Internal knots are typically uniform in parameter space [0, 1]
        # We use linspace(0, 1, num=n_internal_knots)
        internal_knots = np.linspace(0, 1, n_internal_knots)
        
        knots = np.concatenate((
            np.zeros(k),            # k leading zeros for clamping at start
            internal_knots,         # Internal knots (including 0 and 1)
            np.ones(k)              # k trailing ones for clamping at end
        ))
        return knots

    def generate_full_contour(self, xy_quarter):
        """
        Extends a set of 1st quadrant points (xy_quarter) to a full symmetric 
        closed contour using mirror symmetry about the x and y axes.
        The quarter shape must define the boundary from the y-axis (Q1-Q2 interface) 
        to the x-axis (Q1-Q4 interface).
        
        Q1: First Quadrant (Original)
        Q2: Flipped over y-axis
        Q3: Flipped over both axes
        Q4: Flipped over x-axis
        """
        Q1 = xy_quarter.copy()
        # Q2: Flip x. Start from the second point to avoid double counting the point on the y-axis.
        Q2 = np.flipud(np.column_stack([-Q1[:, 0], Q1[:, 1]]))[1:]
        # Q3: Flip x and y. Start from the second point (not on y-axis)
        Q3 = np.column_stack([-Q1[:, 0], -Q1[:, 1]])[1:]
        # Q4: Flip y. Start from the second point (not on x-axis)
        Q4 = np.flipud(np.column_stack([Q1[:, 0], -Q1[:, 1]]))[1:]
        
        # Vertically stack all quadrants
        xy_full = np.vstack([Q1, Q2, Q3, Q4])
        return xy_full



    def evaluate_bspline_contour(self, XY_flat):
        """
        Evaluate a B-spline curve defined by flattened control-point coordinates.
    
        This function reconstructs a 2-D control-point array from the flattened
        input `XY_flat`, builds a clamped knot vector (pre-stored in
        `self.knot_vector`), constructs a B-spline of degree `self.k`, and samples
        the curve at `self.npt` uniformly spaced parameter values.
        
        Parameters
        ----------
        XY_flat : ndarray of shape (N,)
            Flattened control-point coordinates. The required length depends on the
            model’s convention for storing free x- and y-coordinates.
    
        Returns
        -------
        xy : ndarray of shape (self.npt, 2)
            Evaluated B-spline curve points.
    
        XY : ndarray of shape (self.n_cp, 2)
            Reconstructed 2-D control-point coordinate array used to build the spline.
        """
    
        # -----------------------------------------
        # Reconstruct the control-point array XY
        # -----------------------------------------
    
        XY = np.zeros((self.n_cp, 2))
    
        # Fill x-coordinates from the first (n_cp - 1) entries
        XY[0:-1, 0] = XY_flat[0:self.n_cp - 1]
    
        # Fill y-coordinates from the remaining entries
        XY[1:, 1] = XY_flat[self.n_cp - 1:]
    
        # -----------------------------------------
        # Build the B-spline curve
        # -----------------------------------------
    
        t = self.knot_vector                # Clamped knot vector
        spline_r = BSpline(t, XY, self.k)  # B-spline of degree self.k
    
        # Uniform parameter sampling
        u = np.linspace(0, 1, self.npt)
    
        # Evaluate curve
        xy = spline_r(u)
    
        return xy, XY
        
    def rotate_cross_section(self, XY_flat_in, theta):
    
        if XY_flat_in.shape[0]%2 !=0:
            xy_quarter_eroded = self.erode_cross_section(XY_flat_in)
            if xy_quarter_eroded.shape[0]==0:
                XY_flat_in = XY_flat_in[0:-1]
                XY_flat = XY_flat_in.copy()
            else:
                XY_flat = XY_flat_in[:-1].copy()

        else:
            XY_flat = XY_flat_in.copy()
    
        # Evaluate the quarter-shape B-spline
        xy_q, XY_q = self.evaluate_bspline_contour(XY_flat)
        
        # Extend to full shape
        xy_full = self.generate_full_contour(xy_q)
        
        # Optional Smoothing (Post-evaluation Gaussian filter)
        # xy_full = gaussian_filter1d(xy_full, sigma=50.0, axis=0)
        
        # Close the contour for plotting
        # if not np.allclose(xy_full[0], xy_full[-1]):
        #     xy_full = np.vstack([xy_full, xy_full[0]])
            
        x = xy_full[:, 0]
        y = xy_full[:, 1]
        
        
        # Apply rotation to full contour points x_cp, y_cp and x y
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        x_rot = cos_t * x - sin_t * y
        y_rot = sin_t * x + cos_t * y
        # x = x_rot
        # y = y_rot
        xy_final = np.column_stack([x_rot, y_rot])
        
        return xy_final
        
    def erode_cross_section(self, XY_flat_in):
        thick = XY_flat_in[-1]
        XY_flat = XY_flat_in[:-1]
        xy_quarter, XY_q = self.evaluate_bspline_contour(XY_flat)
        L = min(xy_quarter[:,0].max() - xy_quarter[:,0].min(), xy_quarter[:,1].max() - xy_quarter[:,1].min())
        thick_normalized = thick * L

        xy_full = self.generate_full_contour(xy_quarter)
       
        xy_quarter_eroded = np.empty((0, 2))
        
        # Create polygon
        poly = Polygon(xy_full)
    
        # Inward offset (erosion)
        eroded = poly.buffer(-thick_normalized, join_style=2) # type: ignore

        if not eroded.is_empty:
            # Extract eroded boundary
            if isinstance(eroded, Polygon):
                boundary = eroded.exterior
            
                x_new, y_new = boundary.xy # full surface
                # extract quater shape where x>=0 and y>=0
                xy_quarter_eroded = np.column_stack((x_new, y_new))[1::,:]
                mask = (xy_quarter_eroded[:,0]>=0) & (xy_quarter_eroded[:,1]>=0)
                
                xy_quarter_eroded = xy_quarter_eroded[mask, :]
                
                tol = 1e-5
                xy_quarter_eroded = np.round(xy_quarter_eroded / tol) * tol
                # _, idx = np.unique(xy_quarter_eroded, axis=0, return_index=True)
                # xy_quarter_eroded = xy_quarter_eroded[np.sort(idx)]
                xy_quarter_eroded = np.array(
                                        list(dict.fromkeys(map(tuple, xy_quarter_eroded)))
                                    )
                
                xy_quarter_eroded =  np.flipud(xy_quarter_eroded) # making sure order goes from x axis to y axis
                
        return xy_quarter_eroded
    
    def plot_cross_section(self, XY_flat_in, prop, ax=None, 
                        annotateCP=False, addLegend=True, showSymm=False, 
                        addTitle=True, theta=0.0, addFill=False):
        """
        Plots the B-spline cross-section and its control polygon.
        Also calculates and displays the shape properties in the title.
        """
        if XY_flat_in.shape[0]%2 !=0:
            xy_quarter_eroded = self.erode_cross_section(XY_flat_in)
            if xy_quarter_eroded.shape[0]==0:
                XY_flat_in = XY_flat_in[0:-1]
                XY_flat = XY_flat_in.copy()
            else:
                XY_flat = XY_flat_in[:-1].copy()

        else:
            XY_flat = XY_flat_in.copy()
            
        # Evaluate the quarter-shape B-spline
        xy_q, XY_q = self.evaluate_bspline_contour(XY_flat)
        
        # Extend to full shape
        xy_full = self.generate_full_contour(xy_q)
        
        # Optional Smoothing (Post-evaluation Gaussian filter)
        # xy_full = gaussian_filter1d(xy_full, sigma=50.0, axis=0)
        
        # Close the contour for plotting
        # if not np.allclose(xy_full[0], xy_full[-1]):
        #     xy_full = np.vstack([xy_full, xy_full[0]])
            
        x = xy_full[:, 0]
        y = xy_full[:, 1]
        
        # Extend Control Points for full-shape plot
        XY_full = self.generate_full_contour(XY_q)
        x_cp = XY_full[:, 0]
        y_cp = XY_full[:, 1]

        #  Plotting setup
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
    
        
        # Apply rotation to full contour points x_cp, y_cp and x y
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        x_cp_rot = cos_t * x_cp - sin_t * y_cp
        y_cp_rot = sin_t * x_cp + cos_t * y_cp
        # x_cp = x_cp_rot
        # y_cp = y_cp_rot
        
        x_rot = cos_t * x - sin_t * y
        y_rot = sin_t * x + cos_t * y
        # x = x_rot
        # y = y_rot
        xy_final = np.column_stack([x_rot, y_rot])

        # Plot Control Points
        ax.plot(x_cp_rot, y_cp_rot, 'o--', label='Control Points', alpha=0.5)
        # Plot B-Spline Contour
        l=ax.plot(x_rot, y_rot, label=f'B-Spline (k={self.k})', alpha=1.0, linewidth=3.0)
        # print color of this line plot
        line_color = l[0].get_color()
        if addFill:    
            plt.fill(x_rot, y_rot, color=line_color, alpha=0.25)

        if XY_flat_in.shape[0]%2 !=0:
            xy_full_eroded = self.generate_full_contour(xy_quarter_eroded)
            x_eroded = xy_full_eroded[:, 0]
            y_eroded = xy_full_eroded[:, 1]
            x_rot_eroded = cos_t * x_eroded - sin_t * y_eroded
            y_rot_eroded = sin_t * x_eroded + cos_t * y_eroded
            ax.plot(x_rot_eroded, y_rot_eroded, '-', color=line_color, label=f'Inner B-Spline', alpha=0.75)
            if addFill:    
                plt.fill(x_rot_eroded, y_rot_eroded, color='w', alpha=1.0)
                   
        # Optional: show the quarter-shape that was generated
        if showSymm:
            ax.plot(xy_q[:, 0], xy_q[:, 1], 'g-', label=f'Quarter B-Spline', alpha=0.7, linewidth=3.0)
    
        # 8. Annotations and Formatting
        if annotateCP:
            # Only annotate the quarter control points for clarity
            for i in range(len(XY_q[:, 0])):
                ax.text(XY_q[i, 0] * 1.2, XY_q[i, 1] * 1.1, f'$C_{i}$', fontsize=16, ha='right')
        
        if addTitle:
            # Title with calculated properties from prop array
            title_text = (
                r"$\mathbf{1}_{valid}$" + f"={prop[0]:.0f}, "
                + r"$\mathbf{S}$" + f"={prop[1]:.3f}, "
                + r"$\mathbf{A}$" + f"={prop[2]:.3f}" + "\n"
                + r"$\mathbf{J}$" + f"={prop[3]:.3f}, "
                + r"$\mathbf{I_{yy}}$" + f"={prop[4]:.3f}, "
                + r"$\mathbf{I_{zz}}$" + f"={prop[5]:.3f}"
            )
            ax.set_title(title_text, fontdict={'fontsize': 16, 'fontweight': 'medium'})
    
        ax.grid(True)
        limts_value = 1.9
        ax.set_xlim(-limts_value, limts_value)
        ax.set_ylim(-limts_value, limts_value)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        if addLegend:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=16, ncol=1)
        
        plt.tight_layout()
        # Note: If ax is None (i.e., a new figure was created), plt.show() would be here.
        # The original code structure assumes the caller manages the figure display.
        return ax, xy_final

