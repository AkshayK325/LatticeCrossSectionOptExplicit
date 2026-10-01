from meshpy.triangle import MeshInfo, build
import numpy as np
import matplotlib.pyplot as plt
# from skfem import *
from skfem.helpers import grad, dot
from skfem.io.meshio import from_meshio
import meshio
from skfem import ElementTriP1, InteriorBasis, BilinearForm, LinearForm, Functional, solve, enforce
from scipy.ndimage import gaussian_filter1d

from shapely.geometry import LineString

## ⚙️ II. Shape Property Calculations
#-------------------------------------------------------------------------------

class GeomProperties:
    """
    Class to compute geometric properties of a B-spline defined cross-section.
    """

    def __init__(self, spline_geom):
        self.spline_geom = spline_geom
        
    def evaluate_section_metrics(self, XY_flat_in):
        """
        Calculates the full set of metrics for a cross-section defined by the B-spline.
        
        Metrics:
        - O (Open/Closed): 1 if no self-intersection, 0 otherwise.
        - S_f (Sharpness Factor): Mean of (sin(theta/2))^3 for internal control point angles (penalty).
        - A, J, Iy, Iz: Area, Torsional Constant, and Second Moments of Area.
        """
        
        sharp_penalty, Area, J, Iy, Iz = 0.0, 0.0, 0.0, 0.0,0.0

        if XY_flat_in.shape[0]%2 != 0:
            thick = XY_flat_in[-1].copy()
            XY_flat = XY_flat_in[0:-1].copy()
        else:
            XY_flat = XY_flat_in.copy()
            
        # 1. Evaluate B-spline contour and extract control points
        xy_quarter, XY_q = self.spline_geom.evaluate_bspline_contour(XY_flat)

        # 2. Self-Intersection Check (O)
        # Use a coarse evaluation for speed and stability
        # xy_quarter_coarse = np.append(xy_quarter[::(self.spline_geom.npt//50), :], xy_quarter[-1, :].reshape(1, -1), axis=0)
        # Below new method to get approx 20 points
        coarseIndex = np.linspace(0, xy_quarter.shape[0] - 1, 20, dtype=int) # type: ignore
        xy_quarter_coarse = xy_quarter[coarseIndex, :]
        
        intersections = self.detect_self_intersections(xy_quarter_coarse)
        O = 1.0 * (len(intersections) <= 0)  # 1.0 if closed, 0.0 if self-intersecting

        # 3. Torsional Constant (J)
        # Assuming 'solve_prandtl_quadrant' performs an FEA/FDM solve on the coarse geometry
        # of the quarter shape.
        if O == 1.0:
            J, _, _ = self.solve_prandtl_quadrant(xy_quarter_coarse)
            # 4. Area and Second Moments of Area (A, Iy, Iz)
            Area, Iy, Iz = self.calculate_inertia_properties(xy_quarter)
            # 5. Sharpness Penalty (S_f)
            angles = self.calculate_control_point_angles(XY_q)
            # S_f = Mean of (sin(theta/2))^3
            sharp_penalty = np.mean((np.sin(angles / 2.0)**3))
       
            
        # Ensure properties are absolute, especially J, A, Iy, Iz which can be negative
        # if the contour is traced in reverse or has complex geometry/numerical issues.
        Area, J, Iy, Iz = map(np.abs, [Area, J, Iy, Iz])

        if XY_flat_in.shape[0]%2 != 0 and O == 1.0:
            
            xy_quarter_eroded = self.spline_geom.erode_cross_section(XY_flat_in)
            
            min_size = 10
            if xy_quarter_eroded.shape[0] > min_size + 1:
                # print(f"\nshape of the eroded quarter: {xy_quarter_eroded.shape}\n")
                
                # xy_quarter_coarse_eroded = np.append(xy_quarter_eroded[::(self.spline_geom.npt//20), :], xy_quarter_eroded[-1, :].reshape(1, -1), axis=0)
                coarseIndex = np.linspace(0, xy_quarter_eroded.shape[0] - 1, min_size, dtype=int) # type: ignore
                xy_quarter_coarse_eroded = xy_quarter_eroded[coarseIndex, :]
                              
                # J_inner = 0.0
                J_inner, _, _ = self.solve_prandtl_quadrant(xy_quarter_coarse_eroded)
                # 4. Area and Second Moments of Area (A, Iy, Iz)
                Area_inner, Iy_inner, Iz_inner = self.calculate_inertia_properties(xy_quarter_eroded)
                
                Area = Area - Area_inner
                J = J - J_inner
                Iy = Iy - Iy_inner
                Iz = Iz - Iz_inner
                
        return np.array([O, sharp_penalty, Area, J, Iy, Iz])


    def detect_self_intersections(self, xy):
        """
        Finds self-intersections in a 2D curve defined by xy[:,0]=x and xy[:,1]=y 
        using the shapely geometry library.
        
        Note: This is computationally expensive and is typically done on a coarse 
        evaluation of the B-spline.
        
        Returns:
        - intersections (list of tuples): List of (x, y) coordinates for intersections.
        """
        intersections = []
        n = len(xy)
    
        # Iterate over pairs of non-adjacent segments
        for i in range(n - 1):
            seg1 = LineString([xy[i], xy[i+1]])
            for j in range(i + 2, n - 1):
                if j == i - 1:
                    continue # Already handled by inner loop condition
    
                seg2 = LineString([xy[j], xy[j+1]])
                
                if seg1.intersects(seg2):
                    inter = seg1.intersection(seg2)
                    # Check if the intersection is a single point
                    if "Point" in inter.geom_type:
                        intersections.append((inter.x, inter.y)) # type: ignore
        return intersections

    def calculate_inertia_properties(self, xy_quarter):
        """
        Calculates the cross-sectional Area (A) and the second moments of area (Iy, Iz) 
        using the generalized Shoelace Formula (Surveyor's Formula) applied to the 
        full symmetric polygon defined by the B-spline evaluation points.
        
        Iy and Iz are calculated with respect to the centroidal axes (which are x=0 and y=0 
        due to symmetry), assuming the contour is traced counter-clockwise.
        
        Parameters:
        - xy_quarter (ndarray): Contour points of the quarter shape.
        
        Returns:
        - area (float)
        - Iy (float): Second moment of area about the y-axis (Ix in many texts).
        - Iz (float): Second moment of area about the z-axis (Iy in many texts).
        """
        # 1. Extend the quarter contour to the full closed contour
        xy_full = self.spline_geom.generate_full_contour(xy_quarter)
        
        # 2. Apply optional Gaussian smoothing (original code did this here, keeping for consistency)
        # The original code applies the smoothing in plot_cross_section, but it's fine here too.
        # The current definition of BsplineShapeAI calls this function, so the smoothing
        # should be done prior if intended for the property calculation.
        # For now, let's assume the raw evaluated points are used for property calculation,
        # as smoothing affects geometry.
    
        # 3. Close the contour
        if not np.allclose(xy_full[0], xy_full[-1]):
            xy_full = np.vstack([xy_full, xy_full[0]])
            
        x = xy_full[:, 0]
        y = xy_full[:, 1]
        
        x0, y0 = x[:-1], y[:-1]
        x1, y1 = x[1:], y[1:]
        
        # Fundamental term for Shoelace Formula
        cross_product_sum = x0 * y1 - x1 * y0
        
        # Area
        area = 0.5 * np.abs(np.sum(cross_product_sum))
        
        # Second Moment of Area about Z-axis (Iz, or Ixx in some contexts)
        # Integral of y^2 dA = (1/12) * Sum( (x_i y_{i+1} - x_{i+1} y_i) * (y_i^2 + y_i y_{i+1} + y_{i+1}^2) )
        Iy = (1.0 / 12.0) * np.sum(cross_product_sum * (y0**2 + y0 * y1 + y1**2))
        
        # Second Moment of Area about Y-axis (Iy, or Iyy in some contexts)
        # Integral of x^2 dA = (1/12) * Sum( (x_i y_{i+1} - x_{i+1} y_i) * (x_i^2 + x_i x_{i+1} + x_{i+1}^2) )
        Iz = (1.0 / 12.0) * np.sum(cross_product_sum * (x0**2 + x0 * x1 + x1**2))
        
        return area, Iy, Iz


    def calculate_control_point_angles(self, control_points):
        """
        Compute internal angles (in degrees) at each control point of a closed 
        spline loop, typically used as a penalty for sharp corners.
        
        The input control_points (5, 2) is a quarter-shape. It is temporarily extended 
        to a partial loop (Q1, part of Q2, part of Q4) for angle calculation at the 
        internal points (C1, C2, C3).
        
        Returns:
            angles_rad (ndarray): Internal angles at C1, C2, C3, in radians.
        """
        
        # Using the existing logic which seems to handle the symmetry by including reflections
        # for a shape that eventually loops:
        extended_cp = np.vstack([
            [control_points[1,0], -control_points[1,1]], # Reflection of C1 over x-axis (approx C_(-1))
            control_points, 
            [-control_points[-2,0], control_points[-2,1]] # Reflection of C3 over y-axis (approx C_(N+1))
        ])
        
        angles = []
       
        
        # Compute angles for the 3 interior control points C1, C2, C3 (indices 2, 3, 4 in extended_cp)
        for i in range(1, extended_cp.shape[0] - 1): 
            p_prev = extended_cp[i - 1] # C0 for C1, C1 for C2, C2 for C3
            p_curr = extended_cp[i]     # C1, C2, C3
            p_next = extended_cp[i + 1] # C2 for C1, C3 for C2, C4 for C3
    
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
    
            # If vectors are zero, skip (shouldn't happen with random points)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0:
                angles.append(np.pi)
                continue
                
            v1 /= norm_v1
            v2 /= norm_v2
    
            # Angle between vectors
            dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle_rad = np.arccos(dot)
            
            # This gives the exterior angle (0 to pi). For convexity penalty, this is fine.
            angles.append(angle_rad) 
    
        return np.array(angles)
    
    def solve_prandtl_quadrant(self, polygonpts):

        # add 0,0 to the polygonpts
        polygonpts = np.vstack((np.array([0, 0]), polygonpts.reshape((-1,2))))
    
        verts,cells = self.triangulate_polygon(polygonpts)
    
        meshio_mesh = meshio.Mesh(verts, [("triangle", cells)])
        mesh = from_meshio(meshio_mesh)
    
        # Refine the mesh to get a smoother solution
        mesh = mesh.refined(3)
        
        e = ElementTriP1()
        basis = InteriorBasis(mesh, e)
    
        @BilinearForm
        def prandtl_bilinear(u, v, w):
            return dot(grad(u), grad(v))# type: ignore
    
        @LinearForm
        def prandtl_linear(v, w):
            return 2.0 * v
    
        @Functional
        def integral_phi(w):
            return w['u']
    
        A = prandtl_bilinear.assemble(basis)
        b = prandtl_linear.assemble(basis)
    
        # Identify only the outer boundary (exclude symmetry boundaries x=0 or y=0)
        def is_on_symmetry_line(x):
            tol = 1e-12
            return (np.abs(x[0]) < tol) or (np.abs(x[1]) < tol)
    
        def find_fixed_corners(p, tol=1e-12):
          x, y = p
          index_x = np.where(np.isclose(x, 0, atol=tol))[0]
          ymax_indx = index_x[np.argmax(y[index_x])]
          index_y = np.where(np.isclose(y, 0, atol=tol))[0]
          xmax_indx = index_y[np.argmax(x[index_y])]
          return np.array([xmax_indx,ymax_indx]).astype(int)
            
        symmetry_nodes = np.array([i for i, x in enumerate(mesh.p.T) if is_on_symmetry_line(x)])
        outer_bdy_nodes = np.setdiff1d(mesh.boundary_nodes(), symmetry_nodes)
    
        # Find fixed corner nodes
        corner_nodes = find_fixed_corners(mesh.p, tol=1e-12)
        outer_bdy_nodes = np.union1d(outer_bdy_nodes, corner_nodes)
    
        # Apply Dirichlet BC only on outer boundary
        A, b = enforce(A, b, D=outer_bdy_nodes)
    
        # Solve
        phi_sol = solve(A, b)
        phi_func = basis.interpolate(phi_sol) # type: ignore
    
        # Compute integral
        J_partial = 2 * integral_phi.assemble(basis, u=phi_func)
    
        J_full = 4*J_partial
    
        return J_full, phi_sol, mesh

    def triangulate_polygon(self, points):
        mesh_info = MeshInfo()
        mesh_info.set_points(points)
        facets = [(i, (i+1) % len(points)) for i in range(len(points))]
        mesh_info.set_facets(facets)
    
        mesh = build(mesh_info, max_volume=0.5)
        return np.array(mesh.points), np.array(mesh.elements)
    
    def plot_solution(self,mesh,phi,J,fig=None):
        
        if fig is None:
          # --- Improved Plotting ---
          fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        else:
          ax = fig.gca()
    
        # Plot the solution as a colored surface
        ax.tricontourf(mesh.p[0], mesh.p[1], mesh.t.T, phi, levels=20, cmap='viridis') # type: ignore
    
        # Add a color bar
        cbar = fig.colorbar(ax.tricontourf(mesh.p[0], mesh.p[1], mesh.t.T, phi, levels=20, cmap='viridis'), ax=ax)  # type: ignore
        cbar.set_label('Solution value ($\phi$)')
    
        # Plot the mesh outline (optional, for visualization)
        ax.triplot(mesh.p[0], mesh.p[1], mesh.t.T, color='k', lw=0.5, alpha=0.5)
    
        ax.set_title('Prandtl Torsion Problem Solution\n Torsion Constant J = {:.4f}'.format(J))
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.show()