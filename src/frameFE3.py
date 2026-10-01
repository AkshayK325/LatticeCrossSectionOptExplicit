import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import pypardiso
import scipy
# torch.set_default_dtype(torch.float64)
# torch.set_printoptions(precision=16)
import scipy.io
import sys
from mesher import nodeToElement,MeshFrame
from boundaryCondition import boundaryConditionEx
from linearSolve import DifferentiableSparseSolve
from utilFuncs import sparse_submatrix
class FrameFE:
    def __init__(self,meshSetting,bc,AnalysisSettings):
        
        # torch.cuda.set_device(0)
        # # Store the device index in self.device
        # self.device = torch.cuda.current_device()
        # # self.device = 'cpu'
        # print(self.device)
        
        useCPU = True
        if(torch.cuda.is_available() and (useCPU == False) ):
          self.device = torch.device("cuda:0")
          print("Running on GPU")
        else:
          self.device = torch.device("cpu")
          torch.set_num_threads(18)  
          print("Running on CPU\n")
          print("Number of CPU threads PyTorch is using:", torch.get_num_threads())
        
        self.meshSetting=meshSetting
        nodeXY = meshSetting['nodeXY']
        connectivity = meshSetting['connectivity']
        self.AnalysisSettings = AnalysisSettings
        self.Section = AnalysisSettings['Section'] # for plotting
        self.dim = nodeXY.shape[1];
        self.numDOFPerNod = int(3*(self.dim-1))
        # print(self.numDOFPerNod)
        # print(dim)
        # import sys;sys.exit()
        self.bc = bc
        self.numElemsPerBeam = meshSetting['numElemsPerBeam']
        self.elemSize =  meshSetting['elemSize']
        self.mesh = MeshFrame()
        self.ElemType = meshSetting['ElemType']
        # self.nFunCall=0;
        # self.nFunFail=0;
        # self.topFig, self.topAx = plt.subplots()
        self.nodeXYbase = nodeXY
        self.connectivityBase = connectivity
        self.fea_dtype = torch.float64
        self.reInitFun(nodeXY,connectivity,reMesh =True)
        # self.reInitFun(nodeXY,connectivity)
        # self.fig, self.ax = plt.subplots()
    
    def reInitFun(self,nodeXY,connectivity,reMesh = False):
    
        if reMesh:
            # nodeXY, connectivity = self.mesh.refineFrameMesh(nodeXY, connectivity, self.numElemsPerBeam)
            nodeXY, connectivity = self.mesh.refineFrameMeshElemSize(nodeXY, connectivity, self.elemSize)
            
            self.nodeToElem = nodeToElement(connectivity)
            
            ## Note self.nodeXY is torch??
            self.nodeXY, self.numNodes = nodeXY, nodeXY.shape[0]
            self.nodeXYtorch = torch.from_numpy(self.nodeXY).to(self.device)
            self.fea_dtype = self.nodeXYtorch.dtype
            self.ndof = self.numDOFPerNod*self.numNodes
            self.connectivity = connectivity
            self.numEle, self.numDOFPerEle = connectivity.shape[0], int(self.numDOFPerNod*2)
            self.barCenter = 0.5*(nodeXY[connectivity[:,0]] + \
                                  nodeXY[connectivity[:,1]])
            
            A =  self.connectivity;
            A = np.tile(A.T, self.numDOFPerNod).reshape(self.numDOFPerEle,-1).T;    
            B = np.arange(self.numDOFPerNod)  # a 1x3 matrix
            B = np.tile(B, self.numEle*2).reshape(self.numEle,-1)
            self.edofMat = A*self.numDOFPerNod + B
            self.edofMat = np.array(self.edofMat,dtype=int)
            # print(self.edofMat)
            # import sys;sys.exit()
        
            self.iK = np.kron(self.edofMat,np.ones((self.numDOFPerEle ,1))).flatten()
            self.jK = np.kron(self.edofMat,np.ones((1,self.numDOFPerEle))).flatten()
            bK = tuple(np.zeros((len(self.iK))).astype(int)) #batch values
            self.nodeIdx =  np.array([self.iK.astype(int), self.jK.astype(int)])
            
            self.iF = self.edofMat.flatten()
            self.jF = 0*self.edofMat.flatten()
            bF = tuple(np.zeros((len(self.iF))).astype(int)) #batch values
            self.nodeIdxF = np.array([self.iF.astype(int), self.jF.astype(int)])
            self.nodeIdxF = torch.from_numpy(self.edofMat.flatten().astype(int)).to(self.device)  # dense only
            self.ZeroForce = torch.zeros((self.ndof)).to(dtype=self.fea_dtype).to(self.device) 
            self.applyForceOnNode(self.bc['forces'])
            self.Aplt = torch.ones((self.connectivity.shape[0],3)).to(dtype=self.fea_dtype)
            
            
            KelemGeoBase = torch.zeros((12,12)).to(dtype=self.fea_dtype).to(self.device)
            KelemGeoBase[[0,6,3,9],[0,6,3,9]]=1.;KelemGeoBase[[0,6,9,3],[6,0,3,9]]=-1.;
            KelemGeoBase[[1,2,7,8],[1,2,7,8]]=6./5;
            KelemGeoBase[[1,2,7,8],[7,8,1,2]]=-6/5;
            KelemGeoBase[[1,1,5,11,8,4,10,8],[5,11,1,1,4,8,8,10]]=1./10;
            KelemGeoBase[[2,2,4,5,7,10,11,7],[4,10,2,7,5,2,7,11]]= -1./10;
            KelemGeoBase[[4,5,10,11],[4,5,10,11]]=2./15;
            KelemGeoBase[[5,11,4,10],[11,5,10,4]]= -1./30;
            self.KelemGeoBase = KelemGeoBase.repeat(self.numEle,1,1)
            KelemMatBase = torch.tensor([[1.0,0,0,0,0,0,-1.,0,0,0,0,0],
                                 [0,12.,0,0,0,6.0,0,-12.,0,0,0,6.],
                                 [0,0,12.,0,-6.0,0,0,0,-12.,0,-6.,0],
                                 [0,0,0,1.0,0,0,0,0,0,-1.0,0,0],
                                 [0,0,-6.0,0,4.0,0,0,0,6.0,0,2,0],
                                 [0,6.0,0,0,0,4.0,0,-6.0,0,0,0,2],
                                 [-1.0,0,0,0,0,0,1.0,0,0,0,0,0],
                                 [0,-12.0,0,0,0,-6.0,0,12.0,0,0,0,-6.0],
                                 [0,0,-12.0,0,6.0,0,0,0,12.0,0,6.0,0],
                                 [0,0,0,-1.0,0,0,0,0,0,1.0,0,0],
                                 [0,0,-6.0,0,2,0,0,0,6.0,0,4.0,0],
                                 [0,6.0,0,0,0,2,0,-6.0,0,0,0,4.0]]).to(dtype=self.fea_dtype).to(self.device)
            self.KelemMatBase = KelemMatBase.repeat(self.numEle,1,1)
            
          
        # NPdist = np.sqrt((nodeXY[connectivity[:,0]][:,0] - \
        #                            nodeXY[connectivity[:,1]][:,0])**2 + \
        #                           (nodeXY[connectivity[:,0]][:,1] - \
        #                            nodeXY[connectivity[:,1]][:,1])**2 +\
        #                           (nodeXY[connectivity[:,0]][:,2] - \
        #                            nodeXY[connectivity[:,1]][:,2])**2)
                                   
        # self.eleLength = torch.from_numpy(NPdist).to(self.device) 
        
        # Get the node coordinates for the two ends of each element
        node1 = self.nodeXYtorch[self.connectivity[:, 0]]
        node2 = self.nodeXYtorch[self.connectivity[:, 1]]
        
        # Compute the Euclidean distance (L2 norm) between node1 and node2
        self.eleLength = torch.norm(node1 - node2, dim=1)                           
                                   
        self.eleCurLength = self.eleLength.clone();
        # print(self.eleLength)
        # print(NPdist)
        
        ## added
        # Predefine 3 candidate reference vectors
        ref_candidates = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ], dtype=self.fea_dtype).to(self.device)
        
        # Beam directions (normalized)
        d = self.nodeXYtorch[self.connectivity[:, 1]] - self.nodeXYtorch[self.connectivity[:, 0]] # type: ignore
        d_normalized = d / torch.linalg.norm(d, dim=1, keepdim=True)
        
        # Compute dot products with the 3 candidates → absolute value to measure alignment
        dots = torch.abs(torch.matmul(d_normalized, ref_candidates.T))   # shape (Nelements, 3)
        
        # For each element pick the candidate with the SMALLEST dot product
        idx = torch.argmin(dots, dim=1)
        
        # Assign v as that candidate
        self.v = ref_candidates[idx]
            
        # 3 candidate reference vectors
        ref_candidates = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        
        # Beam direction vectors
        d = self.nodeXYbase[self.connectivityBase[:, 1]] - self.nodeXYbase[self.connectivityBase[:, 0]]
        
        # Normalize direction vectors
        d_normalized = d / np.linalg.norm(d, axis=1, keepdims=True)
        
        # Dot products with all 3 candidates
        dots = np.abs(d_normalized @ ref_candidates.T)   # (Nelements x 3)
        
        # For each element choose candidate least aligned with beam direction
        idx = np.argmin(dots, axis=1)
        
        # Assign v_base
        self.v_base = ref_candidates[idx]
                
        
        # # # Choose reference vector v based on predominant axis
        # def choose_reference_vector(axis):
        #     if axis == 0 or axis == 1:  # x-axis or y-axis
        #         return torch.tensor([0, 0, -1.],dtype=self.fea_dtype).to(self.device) 
        #     else:  # z-axis
        #         return torch.tensor([1., 0, 0],dtype=self.fea_dtype).to(self.device) 
        
        # d = torch.sqrt((self.nodeXYtorch[self.connectivity[:, 1]] - self.nodeXYtorch[self.connectivity[:, 0]])**2)
        # # d = self.eleLength.clone()
    
        # # Normalize direction vectors
        # d_normalized = d / torch.linalg.norm(d, dim=1, keepdim=True)
        
        # # Determine predominant axis
        # predominant_axis = torch.argmax(d_normalized, dim=1)
        
        # self.v = torch.stack([choose_reference_vector(axis) for axis in predominant_axis]).to(self.device) 
        
        #### 
        # # Buildup base v for base mesh
        # d = np.sqrt((self.nodeXYbase[self.connectivityBase[:, 1]] - self.nodeXYbase[self.connectivityBase[:, 0]])**2)
        # # d = self.eleLength.clone()
    
        # # Normalize direction vectors
        # d_normalized = d / np.linalg.norm(d, axis=1, keepdims=True)
        
        # # Determine predominant axis
        # predominant_axis = np.argmax(d_normalized, axis=1)
        
        # self.v_base = torch.stack([choose_reference_vector(axis) for axis in predominant_axis]).detach().numpy()
        
        
        
        # x = torch.zeros((self.numEle,3)).to(self.device) 
        # v = torch.zeros_like(x).to(self.device) 
        # v[:,-1]=1
        
        # x[:,0] = self.nodeXYtorch[self.connectivity[:,1]][:,0] - self.nodeXYtorch[self.connectivity[:,0]][:,0]  #dx
        # x[:,1] = self.nodeXYtorch[self.connectivity[:,1]][:,1] - self.nodeXYtorch[self.connectivity[:,0]][:,1]  #dy
        # x[:,2] = self.nodeXYtorch[self.connectivity[:,1]][:,2] - self.nodeXYtorch[self.connectivity[:,0]][:,2]  #dz
        
        # Calculate differences in x, y, z coordinates for all elements dx,dy dz
        x = self.nodeXYtorch[self.connectivity[:, 1]] - self.nodeXYtorch[self.connectivity[:, 0]]
        
        x = x / torch.linalg.norm(x, dim=1, keepdim=True) # torch.einsum('ni,n->ni',x,1./torch.linalg.norm(x,dim=1));
        y = torch.linalg.cross(self.v,x);
        y = y / torch.linalg.norm(y, dim=1, keepdim=True) # torch.einsum('ni,n->ni',y,1./torch.linalg.norm(y,dim=1));
        z = torch.linalg.cross(x,y);
        
        # R = [ x(1) x(2) x(3);
        #       y(1) y(2) y(3);
        #       z(1) z(2) z(3) ];
        
        # self.T = torch.zeros((self.numEle,3,3)).to(self.device) 
        # # 3D rotation matrix calculation el x 3 x3
        # self.T[:,0,0] = x[:,0];self.T[:,1,1] = y[:,1];self.T[:,2,2] = z[:,2];
        # self.T[:,0,1] = x[:,1];self.T[:,0,2] = x[:,2]
        # self.T[:,1,0] = y[:,0]; self.T[:,1,2] = y[:,2];     
        # self.T[:,2,0] = z[:,0]; self.T[:,2,1] = z[:,1]; 
        
        self.T = torch.stack((x, y, z), dim=1)
        
        self.TN1 = self.T.transpose(1,2)
        self.TN2 = self.TN1.clone()
        self.Rm = torch.tile(torch.eye(3,dtype=self.fea_dtype).to(self.device) ,(self.numEle,1,1)).to(self.device) 
        self.RmBase = self.Rm.clone()
        
        self.u_np = self.applyDirichletOnNode(self.bc['fixtures'])

  #--------------------------#
    def applyDirichletOnNode(self, fixed):
        self.fixedDofs = []
        u = np.zeros(self.ndof)
        for i in range(np.shape(fixed['nodes'])[0]):
            if fixed['ux'][i,0] == 1:
                dof = self.numDOFPerNod*fixed['nodes'][i]
                self.fixedDofs = np.append(self.fixedDofs, dof).astype(int)
                u[dof] = fixed['ux'][i,1]
            if fixed['uy'][i,0] == 1:
                dof = self.numDOFPerNod*fixed['nodes'][i]+1
                self.fixedDofs = np.append(self.fixedDofs, dof).astype(int)
                u[dof] = fixed['uy'][i,1]
            if fixed['uz'][i,0] == 1:
                dof = self.numDOFPerNod*fixed['nodes'][i]+2
                self.fixedDofs = np.append(self.fixedDofs, dof).astype(int)
                u[dof] = fixed['uz'][i,1]
            if fixed['utheta1'][i,0] == 1:
                dof = self.numDOFPerNod*fixed['nodes'][i]+3
                self.fixedDofs = np.append(self.fixedDofs, dof).astype(int)
                u[dof] = fixed['utheta1'][i,1]
            if fixed['utheta2'][i,0] == 1:
                dof = self.numDOFPerNod*fixed['nodes'][i]+4
                self.fixedDofs = np.append(self.fixedDofs, dof).astype(int)
                u[dof] = fixed['utheta2'][i,1]
            if fixed['utheta3'][i,0] == 1:
                dof = self.numDOFPerNod*fixed['nodes'][i]+5
                self.fixedDofs = np.append(self.fixedDofs, dof).astype(int)
                u[dof] = fixed['utheta3'][i,1]
            
        # self.fixedDofs = np.append(self.fixedDofs, self.numDOFPerNod*fixed['YNodes']+1).astype(int)
        # self.fixedDofs = np.append(self.fixedDofs, self.numDOFPerNod*fixed['ThetaNodes']+2).astype(int)
        self.freeDofs = np.setdiff1d(np.arange(self.ndof), self.fixedDofs)
        V = np.zeros((self.ndof, self.ndof))
        V[self.fixedDofs,self.fixedDofs] = 1.
        V = torch.tensor(V[np.newaxis])
        indices = torch.nonzero(V).t()
        values = V[indices[0], indices[1], indices[2]]
        self.fixedBCPenaltyMatrix = \
            torch.sparse_coo_tensor(indices, values, V.size()).to(self.device) 
            
        return u
    #--------------------------#
    def applyForceOnNode(self, force):
        self.force = self.ZeroForce.clone().flatten()
        # force = force.to(self.device)
        self.force[np.array([self.numDOFPerNod])*force['nodes']] = torch.tensor(force['fx']).to(self.device)
        self.force[np.array([self.numDOFPerNod])*force['nodes']+1] = torch.tensor(force['fy']).to(self.device)
        self.force[np.array([self.numDOFPerNod])*force['nodes']+2] = torch.tensor(force['fz']).to(self.device)
    
        self.force[np.array([self.numDOFPerNod])*force['nodes']+3] = torch.tensor(force['mx']).to(self.device)
        self.force[np.array([self.numDOFPerNod])*force['nodes']+4] = torch.tensor(force['my']).to(self.device)
        self.force[np.array([self.numDOFPerNod])*force['nodes']+5] = torch.tensor(force['mz']).to(self.device)
    
        # self.force = self.force.unsqueeze(0).unsqueeze(2).to(self.device) 
    # 
    def updateGeo(self,du,u):
        uel = u[self.edofMat.reshape(-1)].reshape([self.numEle,-1]) # get element wise deformation matrix (el x node dof)
        duel= du[self.edofMat.reshape(-1)].reshape([self.numEle,-1]) # get element wise last config deformation matrix (el x node dof)
        
        unode = u.reshape([self.numNodes,-1]); # get nodal deformatin (node num x node dof)
        nodeXYnew = self.nodeXYtorch + unode[:,0:self.dim]; # skip the rot nodes, add dx dy to X and Y
    
        # duel = torch.einsum('nij,nj->ni',self.T,duel) # need to assemble ndof x 1
        # 
        def largeRotMtx(d_r: torch.Tensor, RmBase: torch.Tensor, numEle: int, device: torch.device) -> torch.Tensor:
            r = torch.linalg.norm(d_r,dim=1) + 1e-18; # add 1e-18 to avoid nan sensitivity
            S = torch.zeros((numEle,3,3)).to(dtype=self.fea_dtype).to(device)
            # S[:,0,1] = -d_r[:,2];
            # S[:,0,2] = d_r[:,1]
            # S[:,1,0] = d_r[:,2]; 
            # S[:,1,2] = -d_r[:,0]
            # S[:,2,0] = -d_r[:,1]; 
            # S[:,2,1] = d_r[:,0]
            S[:,[0,1,2],[1,2,0]] = -d_r[:,[2,0,1]];
            S[:,[0,1,2],[2,0,1]] = d_r[:,[1,2,0]];
            
            weights1 = torch.sin(r) / r  # Shape: (n,)
            weights2 = (1 - torch.cos(r)) / r**2  # Shape: (n,)
            S2 = torch.bmm(S, S)           # Shape: (n, i, j)
            
            # Broadcasting for elementwise multiplication:
            term1 = weights1.view(-1, 1, 1) * S  # Shape: (n, i, j)
            term2 = weights2.view(-1, 1, 1) * S2 # Shape: (n, i, j)
            
            # term1 = torch.einsum('n,nij->nij',torch.sin(r)/r , S) # slow
            # term2 = torch.einsum('n,nij->nij',(1-torch.cos(r))/r**2 , S2) # slow
    
            Rt = RmBase + torch.nan_to_num(term1 + term2) # torch.einsum('nij,njk->nik',S,S)
            return Rt
            
        # rm = torch.zeros((self.numEle,3)).to(self.device)    
        # rm[:,0] = duel[:,3]+duel[:,9]
        # rm[:,1] = duel[:,4]+duel[:,10]
        # rm[:,2] = duel[:,5]+duel[:,11]
        # rm=0.5*rm
        rm = (duel[:, 3:6] + duel[:, 9:12])/2
        Rm = largeRotMtx(rm,self.RmBase,self.numEle,self.device)
        RN1 = largeRotMtx(duel[:,3:6],self.RmBase,self.numEle,self.device)
        RN2 = largeRotMtx(duel[:,9:12],self.RmBase,self.numEle,self.device)
        
        # nn= 111
        # print(Rm[nn,:,:],RN1[nn,:,:],RN2[nn,:,:],(RN1[nn,:,:]+RN2[nn,:,:])/2)
        
        # self.Rm = torch.einsum('nij,nkj->nik',Rm,self.T); # slow
        self.Rm = torch.bmm(Rm, self.T.transpose(1, 2))
        
        # x = torch.zeros((self.numEle,3)).to(self.device) 
        # v = torch.zeros_like(x).to(self.device) 
        # v[:,-1]=1.0
        # x[:,0] = (nodeXYnew[self.connectivity[:,1]][:,0] - nodeXYnew[self.connectivity[:,0]][:,0]) #dx
        # x[:,1] = (nodeXYnew[self.connectivity[:,1]][:,1] - nodeXYnew[self.connectivity[:,0]][:,1]) #dy
        # x[:,2] = (nodeXYnew[self.connectivity[:,1]][:,2] - nodeXYnew[self.connectivity[:,0]][:,2]) #dz
        
        x = nodeXYnew[self.connectivity[:, 1]] - nodeXYnew[self.connectivity[:, 0]]
    
        r1 = self.Rm[:,:,0];
        r2 = self.Rm[:,:,1];
        r3 = self.Rm[:,:,2];
        
        x = x / torch.linalg.norm(x, dim=1, keepdim=True) #torch.einsum('ni,n->ni',x,1./torch.linalg.norm(x,dim=1));
        r1x =  torch.sum(r1*x,dim=1) #torch.einsum('ni,ni->n',r1,x) #einsum slower
        r2x =  torch.sum(r2*x,dim=1) #torch.einsum('ni,ni->n',r2,x)
        r3x =  torch.sum(r3*x,dim=1) #torch.einsum('ni,ni->n',r3,x)
        y = r2 - (r2x / (1. + r1x + 1e-12))[:, None] * (r1 + x) # torch.einsum('n,ni->ni',r2x/(1.+r1x+1e-12),(r1+x))
        z = r3 - (r3x / (1. + r1x + 1e-12))[:, None] * (r1 + x) #torch.einsum('n,ni->ni',r3x/(1.+r1x+1e-12),(r1+x));
        y =  y / torch.linalg.norm(y, dim=1, keepdim=True) #torch.einsum('ni,n->ni',y,1./torch.linalg.norm(y,dim=1));
        z =  z / torch.linalg.norm(z, dim=1, keepdim=True) #torch.einsum('ni,n->ni',z,1./torch.linalg.norm(z,dim=1));
        
        # R = [ x(1) x(2) x(3);
        #       y(1) y(2) y(3);
        #       z(1) z(2) z(3) ];
        # self.T = torch.zeros((self.numEle,3,3)).to(self.device) 
        # # 3D rotation matrix calculation el x 3 x3
        # self.T[:,0,0] = x[:,0];self.T[:,1,1] = y[:,1];self.T[:,2,2] = z[:,2];
        # self.T[:,0,1] = x[:,1];self.T[:,0,2] = x[:,2]
        # self.T[:,1,0] = y[:,0]; self.T[:,1,2] = y[:,2];     
        # self.T[:,2,0] = z[:,0]; self.T[:,2,1] = z[:,1]; 
        
        self.T = torch.stack((x, y, z), dim=1)
        
        # self.eleCurLength = torch.sqrt( (nodeXYnew[self.connectivity[:,0]][:,0] - \
        #                       nodeXYnew[self.connectivity[:,1]][:,0])**2 + \
        #                     (nodeXYnew[self.connectivity[:,0]][:,1] - \
        #                       nodeXYnew[self.connectivity[:,1]][:,1])**2 +\
        #                       (nodeXYnew[self.connectivity[:,0]][:,2] - \
        #                       nodeXYnew[self.connectivity[:,1]][:,2])**2) # get new current lengths
        
        # Get the node coordinates for the two ends of each element
        node1 = nodeXYnew[self.connectivity[:, 0]]
        node2 = nodeXYnew[self.connectivity[:, 1]]
        
        # Compute the Euclidean distance (L2 norm) between node1 and node2
        self.eleCurLength = torch.norm(node1 - node2, dim=1)
    
        self.TN1 = torch.bmm(RN1,self.TN1) # torch.einsum('nij,njk->nik',RN1,self.TN1);
        self.TN2 = torch.bmm(RN2,self.TN2) # torch.einsum('nij,njk->nik',RN2,self.TN2);
        
        self.blkDiag = torch.kron(torch.eye(4,4).to(self.device),self.T)
        
        # del nodeXYnew,RN1,RN2
        
        return 0.
  #--------------------------#
  # 
    def intForceVctNonLin(self,u,E,A,I):
    
        G = E / (2*(1 + 0.0))
        
        uel = u[self.edofMat.reshape(-1)].reshape([self.numEle,-1]) # get element wise deformation matrix (el x node dof)
        # Length increment since beginning of analysis
        dL = self.eleCurLength - self.eleLength;
        
        # Element local basis
        x = self.T[:,0,:];
        y = self.T[:,1,:];
        z = self.T[:,2,:];
        
        # % Nodes local basis
        nx1 = self.TN1[:,:,0];
        ny1 = self.TN1[:,:,1];
        nz1 = self.TN1[:,:,2];
        nx2 = self.TN2[:,:,0];
        ny2 = self.TN2[:,:,1];
        nz2 = self.TN2[:,:,2];
    
        # % Relative rotations
        
        # print("TN1",self.TN1.dtype)
        # print("TN2",self.TN2)
        
        # rx  = (torch.asin(0.5*(torch.einsum('ni,ni->n',z,ny2)-torch.einsum('ni,ni->n',y,nz2)))-torch.asin(0.5*(torch.einsum('ni,ni->n',z,ny1)-torch.einsum('ni,ni->n',y,nz1))));
        # ry1 = torch.asin(0.5*(torch.einsum('ni,ni->n',x,nz1)-torch.einsum('ni,ni->n',z,nx1)));
        # rz1 = torch.asin(0.5*(torch.einsum('ni,ni->n',y,nx1)-torch.einsum('ni,ni->n',x,ny1)));
        # ry2 = torch.asin(0.5*(torch.einsum('ni,ni->n',x,nz2)-torch.einsum('ni,ni->n',z,nx2)));
        # rz2 = torch.asin(0.5*(torch.einsum('ni,ni->n',y,nx2)-torch.einsum('ni,ni->n',x,ny2)));
        
        # faster
        rx  = (torch.asin(0.5*(torch.sum(z*ny2,dim=1)-torch.sum(y*nz2,dim=1)))-torch.asin(0.5*(torch.sum(z*ny1,dim=1)-torch.sum(y*nz1,dim=1))));
        ry1 = torch.asin(0.5*(torch.sum(x*nz1,dim=1)-torch.sum(z*nx1,dim=1)));
        rz1 = torch.asin(0.5*(torch.sum(y*nx1,dim=1)-torch.sum(x*ny1,dim=1)));
        ry2 = torch.asin(0.5*(torch.sum(x*nz2,dim=1)-torch.sum(z*nx2,dim=1)));
        rz2 = torch.asin(0.5*(torch.sum(y*nx2,dim=1)-torch.sum(x*ny2,dim=1)));
        
        
        # print("rx",rx[-1],ry1[-1],ry2[-1],rz1[-1],rz2[-1])
        # print(self.eleCurLength.dtype,self.eleLength.dtype)
        N1 = -(E*A[:,0]*dL)/self.eleLength
        N2 = -N1;
        
        T1 = -(G*I[:,0]*rx)/self.eleLength
        T2 = -T1
        
        if self.ElemType == 'EB':
          My1 = 2*E*I[:,1]*(2*ry1+ry2)/self.eleLength;
          Mz1 = 2*E*I[:,2]*(2*rz1+rz2)/self.eleLength;
          My2 = 2*E*I[:,1]*(ry1+2*ry2)/self.eleLength;
          Mz2 = 2*E*I[:,2]*(rz1+2*rz2)/self.eleLength;
        else:
          # Timoshenko parameters
          omz = E * I[:,2] / (G * A[:,1] * self.eleLength * self.eleLength);
          omy = E * I[:,1] / (G * A[:,2] * self.eleLength * self.eleLength);
          niz = 1 + 12 * omz;
          niy = 1 + 12 * omy;
          laz = 1 + 3  * omz;
          lay = 1 + 3  * omy;
          gaz = 1 - 6  * omz;
          gay = 1 - 6  * omy;
          My1 = 2*E*I[:,1]*(2*lay*ry1+gay*ry2)/(niy*self.eleLength);
          Mz1 = 2*E*I[:,2]*(2*laz*rz1+gaz*rz2)/(niz*self.eleLength);
          My2 = 2*E*I[:,1]*(gay*ry1+2*lay*ry2)/(niy*self.eleLength);
          Mz2 = 2*E*I[:,2]*(gaz*rz1+2*laz*rz2)/(niz*self.eleLength);
          
        Qy1 =  (Mz1+Mz2)/self.eleLength;
        Qz1 = -(My1+My2)/self.eleLength;
        Qy2 = -Qy1;
        Qz2 = -Qz1;
                
        # Assemble element internal force vector in local system
        # fi_local = torch.vstack((N1,Qy1,Qz1,T1,My1,Mz1, N2,Qy2,Qz2,T2,My2,Mz2)).T; # slow
        
        # Concatenate tensors along the first dimension and then transpose
        fi_local = torch.cat((N1.unsqueeze(0), Qy1.unsqueeze(0), Qz1.unsqueeze(0), 
                          T1.unsqueeze(0), My1.unsqueeze(0), Mz1.unsqueeze(0), 
                          N2.unsqueeze(0), Qy2.unsqueeze(0), Qz2.unsqueeze(0), 
                          T2.unsqueeze(0), My2.unsqueeze(0), Mz2.unsqueeze(0)), dim=0).T
    
        # print("local",fi_local[-1,:])
        
        # % Update element corner forces needed to be saved in self?
        # elem.Fc = fil;
        # Transform element internal force from local to global system
        # sF = torch.einsum('nji,nj->ni',self.blkDiag,fi_local) # need to assemble ndof x 1; slower
        sF = torch.bmm(self.blkDiag.transpose(1,2),fi_local.unsqueeze(-1)).squeeze(-1) 
    
        # print("local rot ",sF)
        # sF = sF.flatten()
        # fi_global = torch.sparse_coo_tensor(self.nodeIdxF, sF,\
        #                         (1, self.ndof, 1)).to_dense()
                                
        fi_global = torch.zeros(self.ndof).to(dtype=self.fea_dtype).to(self.device)
    
        fi_global = fi_global.index_add_(0,self.nodeIdxF,sF.flatten());#.reshape((1, self.ndof, 1)) # type: ignore
            
        return fi_global,fi_local
      
    #--------------------------#
    def assembleKNonLin(self,du,u,E,A,I):
      
        self.updateGeo(du,u)
        
        Fi_global,Fi_local = self.intForceVctNonLin(u,E,A,I)
        
        Kmat = self.materialStiff(E,A,I);
        Kgeo = self.geometricStiff(Fi_local,E,A,I);
    
        Kelem = Kmat + Kgeo;
        
        sK = torch.bmm(torch.bmm(self.blkDiag.transpose(1, 2), Kelem), self.blkDiag)
    
        Kasm = torch.sparse_coo_tensor(self.nodeIdx, sK.flatten(),(self.ndof, self.ndof)) # type: ignore
        
        return Kasm,Fi_global
  #--------------------------#
    
    def geometricStiff(self,Fi_local,E,A,I):
        G = E / (2*(1 + 0.0)) 
        
        KelemGeo = self.KelemGeoBase.clone()
                
        L  = self.eleCurLength;
        
        # Timoshenko parameters
        omz = E * I[:,2] / (G * A[:,1] * L * L);
        omy = E * I[:,1] / (G * A[:,2] * L * L);
        muz = 1 + 12 * omz;
        muy = 1 + 12 * omy;
        az  = 1 + 20 * omz + 120 * omz * omz;
        ay  = 1 + 20 * omy + 120 * omy * omy;
        bz  = 1 + 15 * omz + 90  * omz * omz;
        by  = 1 + 15 * omy + 90  * omy * omy;
        cz  = 1 + 60 * omz + 360 * omz * omz;
        cy  = 1 + 60 * omy + 360 * omy * omy;
        
        KelemGeo[:,[1,1,1,2,2,2,4,4,5,5,7,7,8,8,10,10,11,11],[5,6,11,5,10,4,2,8,1,7,5,11,4,10,2,8,1,7]] = torch.einsum( 'ni,n->ni',KelemGeo[:,[1,1,1,2,2,2,4,4,5,5,7,7,8,8,10,10,11,11],[5,6,11,5,10,4,2,8,1,7,5,11,4,10,2,8,1,7]],L)
    
        KelemGeo[:,[4,4,5,5,10,10,11,11],[4,10,5,11,4,10,5,11]] = torch.einsum( 'ni,n->ni',KelemGeo[:,[4,4,5,5,10,10,11,11],[4,10,5,11,4,10,5,11]],L**2)
        
        KelemGeo[:,[3,3,9,9],[3,9,3,9]] = torch.einsum( 'ni,n->ni',KelemGeo[:,[3,3,9,9],[3,9,3,9]],I[:,0]/A[:,0])
        
        # Replacing einsum with element-wise multiplication
        # KelemGeo[:,[1,1,1,2,2,2,4,4,5,5,7,7,8,8,10,10,11,11],[5,6,11,5,10,4,2,8,1,7,5,11,4,10,2,8,1,7]] *= L.unsqueeze(1)
        
        # KelemGeo[:,[4,4,5,5,10,10,11,11],[4,10,5,11,4,10,5,11]] *= L.unsqueeze(1)**2
        
        # KelemGeo[:,[3,3,9,9],[3,9,3,9]] *= (I[:,0] / A[:,0]).unsqueeze(1)
        
    
        if self.ElemType == 'TS':
          KelemGeo[:,[1,5,7,11],:] = torch.einsum( 'nij,n->nij',KelemGeo[:,[1,5,7,11],:],1./muz**2)
          KelemGeo[:,[2,4,8,10],:] = torch.einsum( 'nij,n->nij',KelemGeo[:,[2,4,8,10],:],1./muy**2)
          
          # Broadcasting element-wise multiplication over the appropriate slices
          # KelemGeo[:, [1, 5, 7, 11], :] = KelemGeo[:, [1, 5, 7, 11], :] * (1. / muz**2).view(-1, 1, 1)
          # KelemGeo[:, [2, 4, 8, 10], :] = KelemGeo[:, [2, 4, 8, 10], :] * (1. / muy**2).view(-1, 1, 1)
    
          KelemGeo[:,[1,7,1,7],[1,7,7,1]] = torch.einsum( 'ni,n->ni',KelemGeo[:,[1,7,1,7],[1,7,7,1]],az)
          KelemGeo[:,[2,8,2,8],[2,8,8,2]] = torch.einsum( 'ni,n->ni',KelemGeo[:,[2,8,2,8],[2,8,8,2]],ay)
          KelemGeo[:,[4,10],[4,10]] = torch.einsum( 'ni,n->ni',KelemGeo[:,[4,10],[4,10]],by)
          KelemGeo[:,[5,11],[5,11]] = torch.einsum( 'ni,n->ni',KelemGeo[:,[5,11],[5,11]],bz)      
          KelemGeo[:,[4,10],[10,4]] = torch.einsum( 'ni,n->ni',KelemGeo[:,[4,10],[10,4]],cy)
          KelemGeo[:,[5,11],[11,5]] = torch.einsum( 'ni,n->ni',KelemGeo[:,[5,11],[11,5]],cz)
          
          
        KelemGeo = torch.einsum('n,nij->nij',(Fi_local[:,6]/L),KelemGeo)
        
        return KelemGeo
  #--------------------------#
    
    def materialStiff(self,E,A,I):
      
        G = E / (2*(1 + 0.3))
    
        # % Assemble Gent elastic stiffness matrix in local system    
                                 
        KelemMat = self.KelemMatBase.clone()
        
        L = self.eleCurLength
        # print(L[0],I[0,:],A[0,:])
        
        # Timoshenko parameters
        omz = E * I[:,2] / (G * A[:,1] * L * L);
        omy = E * I[:,1] / (G * A[:,2] * L * L);
        muz = 1 + 12 * omz;
        muy = 1 + 12 * omy;
        laz = 1 + 3  * omz;
        lay = 1 + 3  * omy;
        gaz = 1 - 6  * omz;
        gay = 1 - 6  * omy;
        
        KelemMat[:,[0,0,6,6],[0,6,0,6]] = torch.einsum( 'ni,n->ni',KelemMat[:,[0,0,6,6],[0,6,0,6]],E*A[:,0])
    
        KelemMat[:,[1,1,7,7],[1,7,1,7]] = torch.einsum( 'ni,n->ni',KelemMat[:,[1,1,7,7],[1,7,1,7]],E*(I[:,2]/L**2))
        
        KelemMat[:,[1,1,5,5,7,7,11,11],[5,11,1,7,5,11,1,7]] = torch.einsum( 'ni,n->ni',KelemMat[:,[1,1,5,5,7,7,11,11],[5,11,1,7,5,11,1,7]],E*I[:,2]/L)
        
        KelemMat[:,[5,5,11,11],[5,11,5,11]] = torch.einsum( 'ni,n->ni',KelemMat[:,[5,5,11,11],[5,11,5,11]],E*I[:,2])
    
        KelemMat[:,[2,2,8,8],[2,8,2,8]] = torch.einsum( 'ni,n->ni',KelemMat[:,[2,2,8,8],[2,8,2,8]],E*I[:,1]/L**2)
        
        KelemMat[:,[2,2,4,4,8,8,10,10],[4,10,2,8,4,10,2,8]] = torch.einsum( 'ni,n->ni',KelemMat[:,[2,2,4,4,8,8,10,10],[4,10,2,8,4,10,2,8]],E*I[:,1]/L)
    
        KelemMat[:,[4,4,10,10],[4,10,4,10]] = torch.einsum( 'ni,n->ni',KelemMat[:,[4,4,10,10],[4,10,4,10]],E*I[:,1])
    
        KelemMat[:,[3,3,9,9],[3,9,3,9]] = torch.einsum( 'ni,n->ni',KelemMat[:,[3,3,9,9],[3,9,3,9]],G*I[:,0])
        
        if self.ElemType == 'TS':
          KelemMat[:,[1,5,7,11],:] = torch.einsum( 'nij,n->nij',KelemMat[:,[1,5,7,11],:],1./muz)
          KelemMat[:,[2,4,8,10],:] = torch.einsum( 'nij,n->nij',KelemMat[:,[2,4,8,10],:],1./muy)
                
          # Broadcasting element-wise multiplication over the appropriate slices
          # KelemMat[:, [1, 5, 7, 11], :] = KelemMat[:, [1, 5, 7, 11], :] * (1. / muz).view(-1, 1, 1)
          # KelemMat[:, [2, 4, 8, 10], :] = KelemMat[:, [2, 4, 8, 10], :] * (1. / muy).view(-1, 1, 1)
    
          KelemMat[:,[4,10],[4,10]] = torch.einsum( 'ni,n->ni',KelemMat[:,[4,10],[4,10]],lay)
          KelemMat[:,[5,11],[5,11]] = torch.einsum( 'ni,n->ni',KelemMat[:,[5,11],[5,11]],laz)
          KelemMat[:,[4,10],[10,4]] = torch.einsum( 'ni,n->ni',KelemMat[:,[4,10],[10,4]],gay)
          KelemMat[:,[5,11],[11,5]] = torch.einsum( 'ni,n->ni',KelemMat[:,[5,11],[11,5]],gaz)
          
    
        KelemMat  = torch.einsum('n,nij->nij',(1/L),KelemMat)
    
        return KelemMat
  #--------------------------#
    def solveFELin(self, E, A, I):
        
        # A = A.repeat(self.numElemsPerBeam,1).T.reshape(self.numEle,-1);
        # print("Area shape",A.shape)
        # I = I.repeat(self.numElemsPerBeam,1).T.reshape(self.numEle,-1);
        
        self.E, self.Aplt2 = E, A
        self.Aplt = A.clone()
    
        b = self.force.clone()
        
        # Kasm = self.assembleKLin(E, A, I, self.eleLength);
        
        # initialize the solution array u    
        u = torch.zeros(self.ndof,dtype=self.fea_dtype)
        Kasm,_ = self.assembleKNonLin(u,u, E, A, I)
    
        # get part of the displacement load
        duFix = torch.from_numpy(self.u_np).to(dtype=self.fea_dtype)
        
        u = u + duFix
            
        if self.AnalysisSettings['matrixType'] == 'Sparse':
            
            Ksp = Kasm.clone().coalesce()
                  
            free_dof = torch.tensor(self.freeDofs, dtype=torch.long)
            Ksp_free = sparse_submatrix(Ksp, free_dof, free_dof)# Ksp[self.freeDofs,:][:,self.freeDofs]
            
            u_col = u.clone().unsqueeze(1) # Shape changes from (N,) to (N, 1)
            
            # 2. Perform the sparse matrix multiplication
            ku_col = torch.sparse.mm(Ksp, u_col)
            
            # 3. Reshape the result back to a vector (N,)
            ku = ku_col.squeeze(1)
            b_updated = b - ku
            
            b_free = b_updated[free_dof]
            
            if self.AnalysisSettings['solver'] == 'spsolve':
                u_free  = DifferentiableSparseSolve.apply(Ksp_free, b_free, scipy.sparse.linalg.spsolve)
            elif self.AnalysisSettings['solver'] == 'cg':
                u_free  = DifferentiableSparseSolve.apply(Ksp_free, b_free, scipy.sparse.linalg.cg)
            elif self.AnalysisSettings['solver'] == 'pypardiso':
                u_free  = DifferentiableSparseSolve.apply(Ksp_free, b_free, pypardiso.spsolve)
            else:
                print( self.AnalysisSettings['solver'] + " Solver not recognized. Falling back to default pypardiso solver.")
                u_free  = DifferentiableSparseSolve.apply(Ksp_free, b_free, pypardiso.spsolve)

            
            u[self.freeDofs] = u_free # type: ignore
            # update force with the new displacement 
            # b_updated = torch.sparse.mm(Ksp, u.clone().unsqueeze(1)).squeeze(1)

        elif self.AnalysisSettings['matrixType'] == 'Dense':
            K = Kasm.to_dense()
            
            ku = torch.einsum('ij,j->i', K, u.clone())
            b_updated = b - ku
            
            #   K = Kasm.to_dense().squeeze(dim=0)
            b_free = b_updated[self.freeDofs];
            
            K_free = K[self.freeDofs,:][:,self.freeDofs]
            u_free = torch.linalg.solve(K_free, b_free).flatten()
            u[self.freeDofs] = u_free
            # update force with the new displacement 
            # b_updated = torch.einsum('ij,j->i', K, u.clone())
            
        K = Kasm.to_dense()        
        f_final = torch.einsum('ij,j->i', K, u)

        self.u=u.detach().cpu().numpy()
        self.dispX, self.dispY = u[0::self.numDOFPerNod], u[1::self.numDOFPerNod]
        self.nodalDeformation = torch.sqrt(self.dispX**2 + self.dispY**2)
        # du = (self.dispX[self.connectivity[:,1]] - self.dispX[self.connectivity[:,0]])*torch.cos(self.eleOrientation)
        # dv = (self.dispY[self.connectivity[:,1]] - self.dispY[self.connectivity[:,0]])*torch.sin(self.eleOrientation)
        # self.internalForce = (E*A*(du+dv))/(self.eleLength)
        
        # K = Kasm.to_dense().squeeze(dim=0)
        # b = K@u
    
        # plt.figure(1)
        # self.plotStructure('Linear',plotDeformed = True,TrueScale=True,fig=plt.figure(1))
        # plt.show(block=True)
        
        
        return u, f_final
  
  ######################  
    def plotStructure(self, titleStr, plotDeformed = True,TrueScale=False,fig=plt.figure(1),nodeAnnotate=False,elemAnnotate=False,thicknessPlot=False):
        # plt.ion()
        
        # plt.clf()
        fig.clear()
        # Clear any colorbars
        for ax in fig.axes:
            if ax.collections:  # Colorbars are often stored as collections
                ax.remove()
        
        if torch.is_tensor(self.Aplt):
            self.Aplt = np.round(self.Aplt.detach().cpu().numpy(),2)
        
        if len(self.Aplt) == 0:
            print("Using Area value = 0.1")
            self.Aplt = 0.1*torch.ones((self.connectivity.shape[0],3))
        
        # midpoints = (self.nodeXY[self.connectivity[:,0],:] 
        #              + self.nodeXY[self.connectivity[:,1],:]) / 2
        # # print((midpoints[:,2]>6)*1.0)
        # print(midpoints.shape,self.Aplt.shape)
        # alpha = (midpoints[:,2]>6)*1.0
        # alpha = torch.from_numpy(alpha)
        # self.Aplt  = (1-alpha)*self.Aplt[0,:]  + (alpha)*self.Aplt[1,:] 
            
            
        LScale = np.max(self.eleLength.detach().cpu().numpy());
    
        if TrueScale:
          scale =1
        else:
        #   scale = 0.15*LScale/np.max(to_np(self.nodalDeformation))
          scale = 24
    
        # Unit = "SIunit"
        # if Unit == "inches":
        #     unScale = 39.37
        # else:
        #     unScale =1
       
        nodeXY = self.nodeXY.copy()
        if plotDeformed:
            unode = self.u.reshape([self.numNodes,-1]); # get nodal deformatin (node num x node dof)
            nodeXY = self.nodeXY + scale*unode[:,0:self.dim]; # skip the rot nodes, add dx dy to X and Y
    
        # print(self.nodeXY)
        
        self.ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(nodeXY[:,0], nodeXY[:,1], nodeXY[:,2], c='r', marker='o', label='Points')
      
        if nodeAnnotate:
          dx = np.max(self.nodeXY)*0.01
          for i, point in enumerate(self.nodeXY):
            self.ax.text(point[0]+dx, point[1]+dx, point[2]+dx,f'{i}', color='m', fontsize=8, ha='right')
      
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')
        self.ax.set_title('3D Points and Lines')
        
        A_unq = np.unique(np.mean(self.Aplt,axis=1))
        
        # Generate colors using the colormap 
        c_map_name = 'jet'
        colors_plt_1 = plt.get_cmap(c_map_name)(np.linspace(0, 0.5, int(0+len(A_unq)//2),endpoint=False)) 
        colors_plt_2 = plt.get_cmap(c_map_name)(np.linspace(0.5, 1, int(1+len(A_unq)//2))) 
        # Combine the two color arrays 
        colors_plt = np.concatenate((colors_plt_1, colors_plt_2), axis=0)
        
        eleNum = 0
        for conn in self.connectivity:
          points = nodeXY[conn]
          self.ax.plot(points[:, 0], points[:, 1], points[:, 2], c='k')
          if elemAnnotate:
            dx = np.max(self.nodeXY)*0.01
            self.ax.text(np.mean(points[:, 0])+dx, np.mean(points[:, 1])+dx, np.mean(points[:, 2])+dx,f'{eleNum}', color='r', fontsize=10, ha='right')
          if thicknessPlot:
            if self.Section=='circle':
              indxs = np.where(A_unq == np.mean(self.Aplt[eleNum,:]))
              r_value = np.sqrt(np.mean(self.Aplt[eleNum,:])/np.pi)
              self.ax = self.create_cylinder(self.ax, points[0,:], points[1,:], R=r_value, color=colors_plt[indxs],not_v=self.v[eleNum,:].detach().cpu().numpy())  # Adjust radius and color
              # self.ax = self.create_cylinder(self.ax, points[0,:], points[1,:], R=r_value, color='blue',not_v=self.v[eleNum,:].detach().cpu().numpy())  # Adjust radius and color
            elif self.Section == 'rect':
              self.ax = self.create_rectangle(self.ax,  points[0,:], points[1,:], self.Aplt[eleNum,0].detach().cpu().numpy()*0+1., self.Aplt[eleNum,0].detach().cpu().numpy()*0+0.78, color='blue',not_v=self.v[eleNum,:].detach().cpu().numpy())
            else:
                raise ValueError("Unrecognized section type. Cannot proceed with plotting.")

          eleNum = eleNum + 1;
        # for conn in self.connectivity:
        #   points = self.nodeXY[conn]
        #   self.ax.plot(points[:, 0], points[:, 1], points[:, 2], c='k')
        # self.ax.scatter(self.nodeXY[:,0], self.nodeXY[:,1], self.nodeXY[:,2], c='r', marker='o', label='Points')
        
        # if thicknessPlot:
        #   print(self.connectivity)
          # Add a legend
      
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(titleStr)
      
        # Manually set the aspect ratio by adjusting the limits
        # self.ax.set_xlim3d(nodeXY[:, 0].min(), nodeXY[:, 0].max())
        # self.ax.set_ylim3d(nodeXY[:, 1].min(), nodeXY[:, 1].max())
        # self.ax.set_zlim(self.nodeXY[:, 2].min(), self.nodeXY[:, 2].max())
        # self.ax.view_init(0,0,60)
        # self.ax.view_init(90,-90,0) # xy plane
        # self.ax.view_init(elev=90, azim=0) # xy plane
        # self.ax.view_init(elev=0, azim=-90) # xz plane
    
        r_unq = np.round(np.sqrt(A_unq/np.pi),2)
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(c_map_name, len(r_unq)) )
        sm.set_array(r_unq)  # Dummy array for the colorbar
        
        # # Zoom to a specific region by setting the view 
        # self.ax.set_xlim([0, 40]) 
        # self.ax.set_ylim([15, 35]) 
        # self.ax.set_zlim([-1, 12])
        
        # Add colorbar
        cbar = fig.colorbar(sm, ax=self.ax, shrink=1.0, aspect=15, pad=0.15,
        ticks=r_unq) 
        # Change the font size of the ticks 
        cbar.ax.tick_params(labelsize=14)    
        self.ax.set_aspect('equal')
        return self.ax
  # plt.show()
  
    def create_cylinder(self,ax,p0,p1,R,color,not_v):
    
        slices = 20
        origin = np.array([0, 0, 0])
        #axis and radius
        
        #vector in direction of axis
        v = p1 - p0
        #find magnitude of vector
        mag = np.linalg.norm(v)
        #unit vector in direction of axis
        v = v / mag
        # #make some vector not in the same direction as v
        # not_v = np.array([0, 0, 1])
        # if (v == not_v).all() or (v == -not_v ).all():
        #     not_v = np.array([1, 0, 0])
            
        #make vector perpendicular to v
        n1 = np.cross(v, not_v)
        #normalize n1
        n1 /= np.linalg.norm(n1)
        #make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)
        #surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0, mag, slices)
        theta = np.linspace(0, 2 * np.pi, slices)
        #use meshgrid to make 2d arrays
        t, theta = np.meshgrid(t, theta)
        #generate coordinates for surface
        X, Y, Z = [p0[i] + v[i] * t + R.item() * np.sin(theta) * n1[i] + R.item() * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        
        # ax.plot_surface(X, Y, Z,color=color,shade=True)
        
        # Plot the surface with shading
        ax.plot_surface(X, Y, Z, color=color, shade=True)
        
        return ax
    
    def create_rectangle(self, ax, p0, p1, height,width, color,not_v):
        # Vector in the direction of the axis
        v = p1 - p0
        # Find the magnitude of vector
        mag = np.linalg.norm(v)
        # Unit vector in the direction of the axis
        v = v / mag
        
        # # Make some vector not in the same direction as v
        # not_v = np.array([0, 0, -1])
        # if (v == not_v).all() or (v == -not_v ).all():
        #     not_v = np.array([0, 0, -1])
        
        # Make vectors perpendicular to v
        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(v, n1)
        
        # Create the 8 vertices of the rectangular prism
        points = np.array([
            p0 + 0.5*width*n1 + 0.5*height*n2,
            p0 - 0.5*width*n1 + 0.5*height*n2,
            p0 - 0.5*width*n1 - 0.5*height*n2,
            p0 + 0.5*width*n1 - 0.5*height*n2,
            p1 + 0.5*width*n1 + 0.5*height*n2,
            p1 - 0.5*width*n1 + 0.5*height*n2,
            p1 - 0.5*width*n1 - 0.5*height*n2,
            p1 + 0.5*width*n1 - 0.5*height*n2
        ])
        
        # Define the 6 faces of the prism using the vertices
        faces = [
            [points[0], points[1], points[2], points[3]], # bottom face
            [points[4], points[5], points[6], points[7]], # top face
            [points[0], points[3], points[7], points[4]], # side face
            [points[1], points[2], points[6], points[5]], # opposite side face
            [points[0], points[1], points[5], points[4]], # front face
            [points[2], points[3], points[7], points[6]]  # back face
        ]
        
        # Convert base_color to RGB array
        base_rgb = np.array(mcolors.to_rgb(color))
        
        # Adjust colors for lighting effects
        face_colors = [
            base_rgb,                     # bottom face (brightest)
            base_rgb * 0.5,               # top face (darkest)
            base_rgb * 0.7,               # side face 1
            base_rgb * 0.7,               # side face 2
            base_rgb * 0.8,               # front face
            base_rgb * 0.8                # back face
        ]
        
        # Plot the rectangular prism
        # ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1))
         # Plot each face with shadow effect
        for i, face in enumerate(faces):
            ax.add_collection3d(Poly3DCollection([face], facecolors=face_colors[i], linewidths=0.2, edgecolors='r', alpha=.9))
    
    
        return ax
  
    def plot_at_iteration(self, i):
       
        # self.ax.clear()  # Clear the current plot
        self.u = self.u_all[i,:].detach().cpu().numpy() # type: ignore
        self.plotStructure('',plotDeformed = True,thicknessPlot=True,TrueScale=True,fig=self.fig,nodeAnnotate=False)
        self.fig.canvas.draw_idle()

    def create_slider(self,fig=plt.figure(1)):
        # Create a slider
        self.fig = fig;
        # self.ax = self.fig.add_subplot(111,projection='3d')
        plt.subplots_adjust(left=0.1, bottom=0.25)
        ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])# type: ignore
        slider = Slider(ax_slider, '', 0, self.num_iterations - 1, valinit=0, valstep=1) # type: ignore
        self.fig.text(0.5, 0.17, 'Load Step', ha='center', va='center', fontsize=12)
  
        # Update function for the slider
        def update(val):
            i = int(slider.val)
            self.plot_at_iteration(i)
  
        # Connect the update function to the slider
        slider.on_changed(update)
  
        # Initial plot at iteration 0
        self.plot_at_iteration(0)
  
        # Display the plot with slider
        plt.show()
  
  
    def plotDvF(self,ui,Fi):
        plt.ion()
        plt.clf()
        
        N = np.shape(self.nodeXY)
        nNode,dim = N[0],N[1]
        top = np.where(self.nodeXY[:,-1] == np.max(self.nodeXY[:,-1])) #z==max
        dofMax = (np.array((top[0]*3*(dim-1))+dim-1).astype(int),)
        # plt.figure(4)
        plt.plot(-torch.mean(ui[:,dofMax[0]],dim=1).detach().cpu(),-torch.sum(Fi[:,dofMax[0]],dim=1).detach().cpu(), linestyle = 'solid',marker='o',markersize=4,linewidth=3,color = 'black')#np.random.rand(3,)
           
        plt.axis('auto')
        plt.xlabel('Displacement (mm)',fontsize=16)
        plt.ylabel('Reaction force (N)',fontsize=16)
        plt.title('Force vs Displacement',fontsize=20)
        # plt.xlim((0,10.2)) 
        # plt.ylim((0,0.88))
        plt.grid(False)
        plt.pause(0.01)
        # plt.show(block='True')

  #--------------------------#
    def getVolume(self):
        return torch.einsum('i,i->i',self.eleLength, self.Aplt).sum()
  #--------------------------#
