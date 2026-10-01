import numpy as np
import trimesh
from collections import Counter
import triangle as tr

class BeamStructureSTL:
    """Class for building and visualizing beam structures as 3D meshes."""
    
    def __init__(self):
        self.verbose = False
        pass
    
    # ----------- Local coordinate system for a beam -----------
    def local_frame(self, P0, P1, v):
        """Compute local coordinate system (ex, ey, ez) for a beam."""
        ez = P1 - P0
        norm_ez = np.linalg.norm(ez)
        if norm_ez == 0:
            raise ValueError("Zero-length beam detected")
        ez /= norm_ez

        v_proj = v - np.dot(v, ez) * ez
        norm_vp = np.linalg.norm(v_proj)
        if norm_vp < 1e-8:
            if abs(ez[0]) < 0.9:
                ey = np.cross(ez, [1,0,0])
            else:
                ey = np.cross(ez, [0,1,0])
        else:
            ey = v_proj / norm_vp

        ex = np.cross(ey, ez)
        return ex, ey, ez

    # ----------- Triangulate 2D polygon -----------
    def triangulate_2d_polygon(self, points_2d):
        """Triangulate a 2D polygon using Triangle library."""
        n = len(points_2d)
        segments = np.array([[i, (i+1) % n] for i in range(n)])

        tri_input = {
            'vertices': points_2d,
            'segments': segments
        }

        tri_output = tr.triangulate(tri_input, 'pQ')  
        return tri_output['triangles']

    # ----------- Find joint nodes -----------
    def find_joints(self, beams):
        """Find nodes that connect to multiple beams (joints)."""
        flat_nodes = beams.flatten()
        counts = Counter(flat_nodes)
        joint_nodes = {node for node, c in counts.items() if c > 1}
        return joint_nodes

    # ----------- Sweep a beam -----------
    def sweep_beam(self, P0, P1, v, section, extension=0.0, joint_nodes=None, start_idx=None, end_idx=None):
        """Sweep a cross-section along a beam path to create a 3D mesh."""
        dir_vec = P1 - P0
        length = np.linalg.norm(dir_vec)
        if length == 0:
            raise ValueError("Zero-length beam detected")
        dir_unit = dir_vec / length

        if joint_nodes is not None:
            if start_idx in joint_nodes:
                P0 = P0 - length * extension * dir_unit
            if end_idx in joint_nodes:
                P1 = P1 + length * extension * dir_unit
        
        ex, ey, ez = self.local_frame(P0, P1, v)
        S0 = section.reshape(-1, 2)
        n = len(S0)
        
        # Remove duplicate padding points
        unique_mask = np.ones(n, dtype=bool)
        for i in range(n-1):
            if np.allclose(S0[i], S0[i+1]):
                unique_mask[i+1] = False
        S0 = S0[unique_mask]
        n = len(S0)
        
        Rmat = np.vstack([ex, ey, ez]).T
        S0_3d = P0 + S0.dot(Rmat[:, :2].T)
        S1_3d = P1 + S0.dot(Rmat[:, :2].T)

        vertices = np.vstack([S0_3d, S1_3d])
        faces = []

        # Side faces with consistent winding (outward normals)
        for i in range(n):
            a0 = i
            a1 = (i+1) % n
            b0 = a0 + n
            b1 = a1 + n
            faces.append([a0, a1, b1])
            faces.append([a0, b1, b0])

        # End caps with correct winding for outward normals
        cap0 = self.triangulate_2d_polygon(S0)
        cap1 = cap0 + n
        # cap1 = [[i + n for i in face] for face in cap0]
        
        # Bottom cap (P0) - reverse winding for outward normal
        for face in cap0:
            faces.append([face[0], face[2], face[1]])
        
        # Top cap (P1) - normal winding for outward normal
        for face in cap1:
            faces.append([face[0], face[1], face[2]])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        mesh.remove_infinite_values()
        return mesh

    # ----------- Create joint connector -----------
    def create_joint_connector_from_faces(self, joint_pos, connected_beams, meshes, beams, nodes):
        """Create joint connector using convex hull of end face vertices at the joint."""
        joint_vertices_list = []
        for beam_idx in connected_beams:
            beam_mesh = meshes[beam_idx]
            a, b = beams[beam_idx]
            P0 = np.array(nodes[a])
            P1 = np.array(nodes[b])
            
            # Determine which end is at the joint
            if np.allclose(P0, joint_pos):
                reference_point = P0
            else:
                reference_point = P1
            
            # Find vertices closest to the joint end
            distances = np.linalg.norm(beam_mesh.vertices - reference_point, axis=1)
            min_distance = np.mean(distances)
            tolerance = max(1e-3, min_distance * 1.1)
            joint_vertex_indices = np.where(distances <= tolerance)[0]
            
            if self.verbose:
                print(f"  Beam {beam_idx}: Found {len(joint_vertex_indices)} vertices at joint, min_dist={min_distance:.6f}, tolerance={tolerance:.6f}")
            
            if len(joint_vertex_indices) > 0:
                end_vertices = beam_mesh.vertices[joint_vertex_indices]
                joint_vertices_list.extend(end_vertices)
        
        if self.verbose:
            print(f"Total vertices collected at joint {joint_pos}: {len(joint_vertices_list)}")
        
        if len(joint_vertices_list) < 3:
            if self.verbose:
                print(f"Warning: Only {len(joint_vertices_list)} vertices found at joint {joint_pos} - need at least 3 for convex hull")
            return None
        
        all_joint_vertices = np.array(joint_vertices_list)
        if self.verbose:
            print(f"Creating convex hull from {len(all_joint_vertices)} vertices")
        
        try:
            hull_mesh = trimesh.convex.convex_hull(all_joint_vertices)
            if self.verbose:
                print(f"Convex hull created: {len(hull_mesh.vertices)} vertices, {len(hull_mesh.faces)} faces")
            return hull_mesh
        except Exception as e:
            print(f"Error creating convex hull: {e}")
            return None

    # ----------- Build full structure -----------
    def build_structure(self, nodes, beams, sections, orientations, sections_color,
        extension=0.00, stl_filename="structure.stl", visualize=False):
        """Build complete beam structure and save as STL."""
        joint_nodes = self.find_joints(beams)
        meshes = []

        for i, (a, b) in enumerate(beams):
            P0 = np.array(nodes[a])
            P1 = np.array(nodes[b])
            v = np.array(orientations[i])
            if self.verbose:
                print(f"Building beam {i} from node {a} to {b}")
            mesh = self.sweep_beam(P0, P1, v, sections[i],
                              extension=extension,
                              joint_nodes=joint_nodes,
                              start_idx=a,
                              end_idx=b)
            mesh.visual.face_colors = sections_color[i] # type: ignore
            meshes.append(mesh)
        
        joint_method = 'hull'
        # Add joint connectors
        if joint_method and joint_method != 'none':
            joint_beam_map = {node: [] for node in joint_nodes}
            for i, (a, b) in enumerate(beams):
                if a in joint_nodes:
                    joint_beam_map[a].append(i)
                if b in joint_nodes:
                    joint_beam_map[b].append(i)
            
            for joint_idx in joint_nodes:
                joint_pos = nodes[joint_idx]
                connected_beams = joint_beam_map[joint_idx]
                
                if len(connected_beams) > 1:
                    joint_connector = self.create_joint_connector_from_faces(
                        joint_pos, connected_beams, meshes, beams, nodes 
                    )
                    if joint_connector is not None:
                        meshes.append(joint_connector)
                        if self.verbose:
                            print(f"Added joint connector at joint {joint_idx}, volume={joint_connector.volume:.6f}")
        if self.verbose:
            # Check mesh diagnostics
            for i, tri_mesh in enumerate(meshes):
                print(f"Mesh {i}: Watertight={tri_mesh.is_watertight}, Volume={tri_mesh.is_volume}, "
                      f"Actual_volume={tri_mesh.volume:.6f}")
                print(f"  - Euler number: {tri_mesh.euler_number}")
                print(f"  - Number of faces: {len(tri_mesh.faces)}")
                print(f"  - Number of vertices: {len(tri_mesh.vertices)}")
        
        # Combine meshes using boolean union
        if self.verbose:
            print("Attempting boolean union...")
        combined = meshes[0].copy()
        combined_colored = meshes[0].copy()
        
        for i, mesh in enumerate(meshes[1:], 1):
            try:
                if self.verbose:
                    print(f"Unioning mesh {i}...")
                result = trimesh.boolean.union([combined, mesh])
                combined_colored = trimesh.util.concatenate([combined_colored, mesh])
                if result is not None:
                    if isinstance(result, list):
                        combined = result[0]
                    else:
                        combined = result
                    if self.verbose:
                        print(f"  Success: {len(combined.vertices)} vertices, {len(combined.faces)} faces")
                else:
                    print(f"  Boolean failed, using concatenation fallback")
                    combined = trimesh.util.concatenate([combined, mesh])
                    
            except Exception as e:
                print(f"  Boolean union failed: {e}")
                print(f"  Using concatenation fallback for mesh {i}")
                combined = trimesh.util.concatenate([combined, mesh])
        
        # Final cleanup
        combined.merge_vertices()
        combined.update_faces(combined.nondegenerate_faces())
        
        # Save as STL
        combined.export(stl_filename)
        print(f"STL saved as: {stl_filename}")
        
        if self.verbose:
            # Check final mesh
            print(f"Final combined mesh: Watertight={combined.is_watertight}, "
                  f"Volume={combined.is_volume}, Actual_volume={combined.volume:.6f}")
        
        if visualize:
            self.load_and_view_stl(stl_filename)
        
        return combined, combined_colored

    # ----------- Load and view STL file -----------
    def load_and_view_stl(self, stl_filename):
        """Load and view STL file with trimesh."""
        try:
            mesh = trimesh.load(stl_filename)
            print(f"\nLoaded STL: {stl_filename}")
            print(f"  Vertices: {len(mesh.vertices)}") # type: ignore
            print(f"  Faces: {len(mesh.faces)}") # type: ignore
            print(f"  Watertight: {mesh.is_watertight}") # type: ignore
            print(f"  Volume: {mesh.volume:.6f}") # type: ignore
            print(f"  Bounds: {mesh.bounds}")
            mesh.show()

            return mesh
        except Exception as e:
            print(f"Error loading/viewing STL: {e}")
            print("Make sure pyglet is installed: pip install pyglet")
            return None


# ----------- Example usage -----------
if __name__ == "__main__":
    beam_struct = BeamStructureSTL()
    
    exmpl = 4
    if exmpl == 1:
        nodes = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=float)
    
        beams = np.array([
            [0, 1],
            [1, 2],
            [2, 3]
        ])
    
        orientations = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=float)
        
        sections = 0.1*np.array([
                [-0.50, -0.50,  0.50, -0.50,  0.50,  0.50, -0.50,  0.50],
                [-0.30, -0.20,  0.30, -0.20,  0.30,  0.20, -0.30,  0.20],
                [-0.40, -0.10,  0.40, -0.10,  0.40,  0.10, -0.40,  0.10]
            ])
    if exmpl == 2:
        nodes = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1.0]
        ], dtype=float)
    
        beams = np.array([
            [0, 3],
            [1, 3],
            [2, 3]
        ])
    
        orientations = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ], dtype=float)
        
        sections = 0.1*np.array([
                [-0.50, -0.50,  0.50, -0.50,  0.50,  0.50, -0.50,  0.50],
                [-0.30, -0.20,  0.30, -0.20,  0.30,  0.20, -0.30,  0.20],
                [-0.40, -0.10,  0.40, -0.10,  0.40,  0.10, -0.40,  0.10]
            ])
    if exmpl == 3:
        nodes = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ], dtype=float)
    
        beams = np.array([
            [0, 3],
            [1, 3],
            [2, 3]
        ])
    
        orientations = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ], dtype=float)
        
        sections = 0.1*np.array([
                [-0.50, -0.50,  0.50, -0.50,  0.50,  0.50, -0.50,  0.50],
                [-0.30, -0.20,  0.30, -0.20,  0.30,  0.20, -0.30,  0.20],
                [-0.40, -0.10,  0.40, -0.10,  0.40,  0.10, -0.40,  0.10]
            ])
    if exmpl == 4:
        nodes = np.array([[ 0.,  0.,  0.],
        [ 0.,  0., 30.],
        [30.,  0., 30.],
        [30., 30., 30.]])
        beams = np.array([[0, 1],
        [1, 2],
        [2, 3]])
        orientations = np.array([[ 1.,  0.,  0.],
        [ 0.,  0., -1.],
        [ 0.,  0., -1.]], dtype=float)
        
        sections = np.array([[ 1.0855e+00,  1.0198e+00,  8.5903e-01,  1.2415e+00,  6.4548e-01,
          1.4001e+00,  4.4485e-01,  1.4957e+00,  2.5238e-01,  1.5351e+00,
          5.0840e-02,  1.5429e+00, -1.6134e-01,  1.5214e+00, -3.8340e-01,
          1.4703e+00, -6.0128e-01,  1.3852e+00, -8.0965e-01,  1.2645e+00,
         -1.0085e+00,  1.1081e+00, -1.1835e+00,  9.2205e-01, -1.3234e+00,
          7.2374e-01, -1.4282e+00,  5.1372e-01, -1.4988e+00,  2.9503e-01,
         -1.5390e+00,  8.3007e-02, -1.5492e+00, -1.2023e-01, -1.5289e+00,
         -3.1511e-01, -1.4573e+00, -5.1824e-01, -1.3236e+00, -7.3820e-01,
         -1.1278e+00, -9.7499e-01, -8.9924e-01, -1.2059e+00, -6.8334e-01,
         -1.3759e+00, -4.8037e-01, -1.4830e+00, -2.8789e-01, -1.5305e+00,
         -8.8276e-02, -1.5437e+00,  1.2197e-01, -1.5275e+00,  3.4277e-01,
         -1.4819e+00,  5.6238e-01, -1.4033e+00,  7.7247e-01, -1.2891e+00,
          9.7305e-01, -1.1392e+00,  1.1544e+00, -9.5685e-01,  1.3005e+00,
         -7.6067e-01,  1.4117e+00, -5.5278e-01,  1.4883e+00, -3.3452e-01,
          1.5339e+00, -1.2090e-01,  1.5496e+00,  8.3932e-02,  1.5352e+00,
          2.7998e-01,  1.4749e+00,  4.8005e-01,  1.3525e+00,  6.9695e-01,
          1.1680e+00,  9.3068e-01],
        [ 1.4894e+00,  3.7667e-03,  1.4753e+00,  3.2032e-01,  1.4276e+00,
          5.8202e-01,  1.3463e+00,  7.8886e-01,  1.2326e+00,  9.4908e-01,
          1.0908e+00,  1.0924e+00,  9.2105e-01,  1.2216e+00,  7.2392e-01,
          1.3359e+00,  5.0665e-01,  1.4225e+00,  2.7201e-01,  1.4766e+00,
          1.9983e-02,  1.4982e+00, -2.3497e-01,  1.4818e+00, -4.7253e-01,
          1.4324e+00, -6.9255e-01,  1.3506e+00, -8.9348e-01,  1.2391e+00,
         -1.0676e+00,  1.1116e+00, -1.2138e+00,  9.7007e-01, -1.3321e+00,
          8.1385e-01, -1.4185e+00,  6.1654e-01, -1.4710e+00,  3.6455e-01,
         -1.4897e+00,  5.7869e-02, -1.4804e+00, -2.6685e-01, -1.4388e+00,
         -5.3852e-01, -1.3636e+00, -7.5533e-01, -1.2555e+00, -9.2149e-01,
         -1.1186e+00, -1.0674e+00, -9.5397e-01, -1.1992e+00, -7.6156e-01,
         -1.3167e+00, -5.4745e-01, -1.4092e+00, -3.1596e-01, -1.4692e+00,
         -6.7098e-02, -1.4967e+00,  1.8990e-01, -1.4873e+00,  4.3064e-01,
         -1.4438e+00,  6.5385e-01, -1.3679e+00,  8.5883e-01, -1.2607e+00,
          1.0380e+00, -1.1358e+00,  1.1893e+00, -9.9684e-01,  1.3127e+00,
         -8.4385e-01,  1.4053e+00, -6.5648e-01,  1.4640e+00, -4.1443e-01,
          1.4888e+00, -1.1770e-01],
        [ 1.3189e-01,  7.8377e-04,  1.7346e-01,  1.1598e-01,  2.9248e-01,
          2.0354e-01,  4.8894e-01,  2.6347e-01,  7.2638e-01,  3.1985e-01,
          8.7314e-01,  4.5975e-01,  9.1731e-01,  6.9103e-01,  8.5864e-01,
          1.0065e+00,  6.9236e-01,  1.2712e+00,  4.1667e-01,  1.4341e+00,
          3.1549e-02,  1.4951e+00, -3.7202e-01,  1.4484e+00, -6.6588e-01,
          1.3006e+00, -8.4911e-01,  1.0522e+00, -9.2308e-01,  7.3198e-01,
         -8.9462e-01,  4.8441e-01, -7.6466e-01,  3.2943e-01, -5.3616e-01,
          2.6504e-01, -3.2499e-01,  2.1264e-01, -1.9092e-01,  1.3170e-01,
         -1.3396e-01,  2.2194e-02, -1.6015e-01, -9.7091e-02, -2.6508e-01,
         -1.8968e-01, -4.4746e-01, -2.5463e-01, -6.8867e-01, -3.0423e-01,
         -8.5408e-01, -4.2752e-01, -9.1691e-01, -6.4219e-01, -8.7712e-01,
         -9.4743e-01, -7.3073e-01, -1.2307e+00, -4.7493e-01, -1.4120e+00,
         -1.0971e-01, -1.4915e+00,  3.0671e-01, -1.4645e+00,  6.2068e-01,
         -1.3349e+00,  8.2403e-01, -1.1048e+00,  9.1736e-01, -7.8695e-01,
          9.0734e-01, -5.2253e-01,  7.9584e-01, -3.5072e-01,  5.8284e-01,
         -2.7150e-01,  3.5765e-01, -2.2429e-01,  2.0956e-01, -1.4854e-01,
          1.3858e-01, -4.4227e-02]
          ], dtype=float)
    
    volumeClose = False
    extension = -0.025
    i = 0
    beam_struct.verbose = True
    while not volumeClose:
        mesh = beam_struct.build_structure(nodes, beams, sections, orientations,
            extension=extension, stl_filename="structure_smooth.stl", visualize=False) # type: ignore
        
        volumeClose = mesh.is_volume
        extension = extension * 0.01
        i = i + 1.0
        if i > 0:
            break
    
    beam_struct.load_and_view_stl("structure_smooth.stl")
    
    # Load and view the generated STL with pyvista
    # import pyvista as pv
    # stl_mesh = pv.read("structure_smooth.stl")
    # plotter = pv.Plotter()
    # plotter.add_mesh(stl_mesh, color='lightblue', show_edges=True)
    # plotter.show()

