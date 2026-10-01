
import torch
from scipy.optimize import Bounds
import numpy as np
import matplotlib.pyplot as plt

from mesher import MeshFrame
from boundaryCondition import boundaryConditionEx
from utilFuncs import custom_minimize
# torch.autograd.set_detect_anomaly(True)



def refineProperty_to_elemSize(Prop_Value,pow_n, nodeXY, connectivity, elemSize, varType='Element'):
    numBeams = connectivity.shape[0]
    numNodes = nodeXY.shape[0]
    
    if varType == 'Element':
        Prop_Value_Base = Prop_Value.clone().reshape((numBeams,-1))
    elif varType == 'Node':
        Prop_Value_Base = Prop_Value.clone().reshape((numNodes,-1))
    elif varType == 'Node+':
        Prop_Value_Base = Prop_Value.clone().reshape((numNodes + numBeams,-1))
    
    node1 = torch.from_numpy(nodeXY[connectivity[:, 0]])
    node2 = torch.from_numpy(nodeXY[connectivity[:, 1]])
    
    # Compute the Euclidean distance (L2 norm) between node1 and node2
    beamLength =  torch.linalg.norm(node1 - node2, dim=1)                           
    
    numElemsPerBeam = torch.round(beamLength / torch.tensor([elemSize], dtype=torch.float64)).int()            
    
    numNodesPerBeam = numElemsPerBeam + 1
    
    Prop_Value_refined = torch.zeros((int(torch.sum(numElemsPerBeam)), Prop_Value_Base.shape[1]), dtype=Prop_Value_Base.dtype) # type: ignore
    n = 0
    for i in range(numBeams):
        
        if varType == 'Element':
            Prop_Value_refined[n:n+numElemsPerBeam[i],:] = Prop_Value_Base[i,:].unsqueeze(0).repeat(int(numElemsPerBeam[i]), 1)
        else:
            t_node = torch.linspace(-1.,1.,numNodesPerBeam[i],dtype=Prop_Value_Base.dtype) # type: ignore
            t_elem = (t_node[:-1] + t_node[1:]) / 2
            
            # quadratic shape
            N0 = t_elem * (t_elem - 1.)/2
            N1 = (1. + t_elem)*(1. - t_elem)
            N2 = t_elem * (t_elem + 1.)/2
            
            P0 = Prop_Value_Base[connectivity[i,0],:];
            P2 = Prop_Value_Base[connectivity[i,1],:];
            P1 = ((P0**(1/pow_n)+P2**(1/pow_n))/2)**pow_n # mid point
    
            if varType == 'Node+':
                P1 = Prop_Value_Base[numNodes+i,:]; # extra mid node
        
            # print(N0*P0[0]+N1*P1[0]+N2*P2[0],N0+N1+N2,P0[0],P1[0],P2[0])
            # print(f"sizes of N0,P0,P1,P2: {N0.shape}, {P0.shape}, {P1.shape}, {P2.shape}")
            
            Prop_Value_refined[n:n+numElemsPerBeam[i],:] = (torch.einsum('i,j->ij',N0,P0**(1/pow_n)) + \
                                                    torch.einsum('i,j->ij',N1,P1**(1/pow_n)) + \
                                                    torch.einsum('i,j->ij',N2,P2**(1/pow_n)))**pow_n # N0*P0+N1*P1+N2*P2
                    

        n += numElemsPerBeam[i]
    
    return Prop_Value_refined
    

def refineVar_to_elemSize(varBase_in, nodeXY, connectivity, elemSize, varType='Element'):
    numBeams = connectivity.shape[0]
    numNodes = nodeXY.shape[0]
    
    if varType == 'Element':
        varBase = varBase_in.clone().reshape((numBeams,-1))
    elif varType == 'Node':
        varBase = varBase_in.clone().reshape((numNodes,-1))
    elif varType == 'Node+':
        varBase = varBase_in.clone().reshape((numNodes + numBeams,-1))
    
    node1 = torch.from_numpy(nodeXY[connectivity[:, 0]])
    node2 = torch.from_numpy(nodeXY[connectivity[:, 1]])
    
    # Compute the Euclidean distance (L2 norm) between node1 and node2
    beamLength =  torch.linalg.norm(node1 - node2, dim=1)                           
    
    numElemsPerBeam = torch.round(beamLength / torch.tensor([elemSize], dtype=torch.float64)).int()            
    
    numNodesPerBeam = numElemsPerBeam + 1
    
    var_refined = torch.zeros((int(torch.sum(numElemsPerBeam)), varBase.shape[1]), dtype=varBase.dtype) # type: ignore
    n = 0
    for i in range(numBeams):
        
        if varType == 'Element':
            var_refined[n:n+numElemsPerBeam[i],:] = varBase[i,:].unsqueeze(0).repeat(int(numElemsPerBeam[i]), 1)
        else:
            t_node = torch.linspace(-1.,1.,numNodesPerBeam[i],dtype=varBase.dtype) # type: ignore
            t_elem = (t_node[:-1] + t_node[1:]) / 2
            
            # quadratic shape
            N0 = t_elem * (t_elem - 1.)/2
            N1 = (1. + t_elem)*(1. - t_elem)
            N2 = t_elem * (t_elem + 1.)/2
            
            P0 = varBase[connectivity[i,0],:];
            P2 = varBase[connectivity[i,1],:];
            P1 = (P0+P2)/2
    
            if varType == 'Node+':
                P1 = varBase[numNodes+i,:]; # extra mid node
        
            # print(N0*P0[0]+N1*P1[0]+N2*P2[0],N0+N1+N2,P0[0],P1[0],P2[0])
            # print(f"sizes of N0,P0,P1,P2: {N0.shape}, {P0.shape}, {P1.shape}, {P2.shape}")
            
            var_refined[n:n+numElemsPerBeam[i],:] = torch.einsum('i,j->ij',N0,P0) + \
                                                    torch.einsum('i,j->ij',N1,P1) + \
                                                    torch.einsum('i,j->ij',N2,P2) # N0*P0+N1*P1+N2*P2
                    

        n += numElemsPerBeam[i]
    
    return var_refined

                        
class OptimizeFrame:

    def __init__(self,frameFE,matProp,exampleInfo,varSetup):
        self.frameFE=frameFE;
        self.matProp = matProp
        self.mesh = MeshFrame()
        self.exampleInfo = exampleInfo
        self.iterations = []
        self.objective_values = []
        self.constraint_values = []
        self.con = 0.
        N = np.shape(self.frameFE.nodeXY)
        nNode,dim = N[0],N[1]
        top = np.where(self.frameFE.nodeXY[:,-1] == np.max(self.frameFE.nodeXY[:,-1])) #z==max only
        self.vf = 0.5
        self.mf = 0.5
        
        self.dofMax = (np.array((top[0]*dim*(dim-1))+dim-1).astype(int),)
        
        self.frameFE.bc = boundaryConditionEx(self.exampleInfo['No'], self.frameFE.nodeXY,self.frameFE.connectivity)
        self.frameFE.Section = varSetup['Section']
        self.varSetup = varSetup
        if varSetup['symArray'] is None:            
            if self.varSetup['varType'] == 'Element':
                self.varSetup['symArray'] = np.arange(frameFE.meshSetting['numUnitLatElem']).tolist() # type: ignore
            if self.varSetup['varType'] == 'Node':
                self.varSetup['symArray'] = np.arange(frameFE.meshSetting['numUnitLatNode']).tolist()  # type: ignore
                
                            
        self.alpha = 0.0 # control the opt phase
        self.obj0 = np.array([1.]); # to normalize the optimization wrt initial guess objective value
        try:
            self.target_dtype = self.frameFE.nodeXYtorch.dtype
        except Exception:
            self.target_dtype = None
    
    def objectiveCall(self,x):
        x = x.reshape((-1,1))
        
        # if not hasattr(self, 'memo'):
        #     self.memo = {}
        
        # see if x is in memo, if yes return obj, sens from memo, else save to memo
        key = tuple(x.flatten().tolist())
        if key in self.memo:
            objValue = self.memo[key]['objValue']
            Sensitivity = self.memo[key]['Sensitivity']
            return objValue,Sensitivity
        
        if self.ObjType == 'SE':
            objValue,Sensitivity = self.objectiveSE(x,Jac=True)
            self.memo[key] = {'objValue': objValue, 'Sensitivity': Sensitivity}
        elif self.ObjType == 'MSEbySE':
            objValue,Sensitivity = self.objectiveMSEbySE(x,Jac=True)
            self.memo[key] = {'objValue': objValue, 'Sensitivity': Sensitivity}
        else:
            raise ValueError("Objective type not recognized.")


        return objValue,Sensitivity
        
    def applySymm(self,x):
        # var = torch.zeros_like(x)
        var = x.clone()
        n_var_col = x.shape[1]
        n_var_row = x.shape[0]
        for i in range(len(self.varSetup['symArray'])):
            a_x = x[self.varSetup['symArray'][i],0:n_var_col].reshape(-1,n_var_col)
            var[self.varSetup['symArray'][i],0:n_var_col] = torch.mean(a_x, dim=0, keepdim=True)
            
        # apply symm to all the unit cell in a lattice
        if self.varSetup['varType'] == 'Element':
            if n_var_row < len(self.frameFE.meshSetting['radiiElemIndex']):
                var = var[self.frameFE.meshSetting['radiiElemIndex'],:]
        elif self.varSetup['varType'] == 'Node':
            if n_var_row < len(self.frameFE.meshSetting['radiiNodIndex']):
                var = var[self.frameFE.meshSetting['radiiNodIndex'],:]
        else:
            varNod = var[0:self.frameFE.meshSetting['numUnitLatNode']]
            varElem = var[self.frameFE.meshSetting['numUnitLatNode']::]
            varNod = varNod[self.frameFE.meshSetting['radiiNodIndex'],:]
            varElem = varElem[self.frameFE.meshSetting['radiiElemIndex'],:]
            var = torch.cat((varNod,varElem))
        
        return var
    
    def apply_var_denormalize(self,var_in):
        # var_in will array of 1D, torch
     
        var_in_denorm = self.varSetup['min_x'] + (self.varSetup['max_x']-self.varSetup['min_x'])*var_in # to save loss of precision
        
        if self.varSetup['symArray'] is not None:
            if self.frameFE.Section =="circle":
                var = self.applySymm(var_in_denorm.reshape((-1,1)))
            if self.frameFE.Section == "z":
                var = self.applySymm(var_in_denorm.reshape((-1,4)))        
        else:
            var = var_in_denorm.clone()
            
        return var
        
    def varToProp(self,var):
        # var of size N,n, with n=1 for circle, and 4 for z, N being number of beams
        
        if self.frameFE.Section =="circle":
            Ax = torch.pi*var.reshape(-1)**2
            Iy = (torch.pi*var.reshape(-1)**4)/4
            A = torch.einsum('i,j->ij',Ax,torch.tensor([1.,1.,1.])) #1e-2 # Ax, Ay, Az (normal, shear, shear)
            I = torch.einsum('i,j->ij',Iy,torch.tensor([2.,1.,1.])) #1e-5 # J1, Iyy, Izz
            S_i = Ax * 0.0 + 1.0 # dummy
        if self.frameFE.Section == "z":
            SAJI = self.varSetup['propPredict'](var)
            S_i = SAJI[:,0]
            Ax = SAJI[:,1]
            A = torch.einsum('i,j->ij',Ax,torch.tensor([1.,1.,1.])) #1e-2 # Ax, Ay, Az (normal, shear, shear)
            I = SAJI[:,2::]
            
        return S_i, A, I
        
    def objectiveSE(self,x_in,Jac=True):
        x = torch.tensor(x_in,requires_grad=True,dtype=self.target_dtype)
        
        var = self.apply_var_denormalize(var_in=x)
        
        _, A, I = self.varToProp(var)
        
        A_fine = refineProperty_to_elemSize(A,torch.tensor([2.0]),self.frameFE.nodeXYbase,self.frameFE.connectivityBase,self.frameFE.elemSize,self.varSetup['varType'])
        I_fine = refineProperty_to_elemSize(I,torch.tensor([4.0]),self.frameFE.nodeXYbase,self.frameFE.connectivityBase,self.frameFE.elemSize,self.varSetup['varType'])

        # var_fine = refineVar_to_elemSize(var,self.frameFE.nodeXYbase,self.frameFE.connectivityBase,self.frameFE.elemSize,self.varSetup['varType'])
        # S_i, A, I = self.varToProp(var_fine)
        # print(A)
        # print(I)
        # A0 = refineVarElemSize(A[:,0],self.frameFE.nodeXYbase,self.frameFE.connectivityBase,self.frameFE.elemSize)
        # A1 = refineVarElemSize(A[:,1],self.frameFE.nodeXYbase,self.frameFE.connectivityBase,self.frameFE.elemSize)
        # A2 = refineVarElemSize(A[:,2],self.frameFE.nodeXYbase,self.frameFE.connectivityBase,self.frameFE.elemSize)
        # I0 = refineVarElemSize(I[:,0],self.frameFE.nodeXYbase,self.frameFE.connectivityBase,self.frameFE.elemSize)
        # I1 = refineVarElemSize(I[:,1],self.frameFE.nodeXYbase,self.frameFE.connectivityBase,self.frameFE.elemSize)
        # I2 = refineVarElemSize(I[:,2],self.frameFE.nodeXYbase,self.frameFE.connectivityBase,self.frameFE.elemSize)
        
        # A_fine = torch.vstack([A0,A1,A2]).T
        # I_fine = torch.vstack([I0,I1,I2]).T
        
        # A_fine = A.clone()
        # I_fine = I.clone()
        
        u,b = self.frameFE.solveFELin(self.matProp['E'].to(A_fine.dtype),A_fine,I_fine)
        # print(torch.min(u),torch.max(u))
        # # whats the location of min u
        # loc = torch.argmin(u) 
        # print(b[loc])
        # print(torch.min(b),torch.max(b))
        objValueSE =0.5*(b*u).sum()
        
        objValue = objValueSE/torch.from_numpy(self.obj0)
    
        if Jac:
            # objValue.backward(retain_graph=True)    
            Sensitivity = torch.autograd.grad(objValue,x,retain_graph=True, create_graph=True)[0].detach().cpu().numpy()
        else:
            Sensitivity = 0*x.detach().cpu().numpy()
        
        objValue = objValue.detach().cpu().numpy()
        
        return objValue, Sensitivity
    
    def objectiveMSEbySE(self,x_in,Jac=True):
        x = torch.tensor(x_in,requires_grad=True,dtype=self.target_dtype)
        
        var = self.apply_var_denormalize(var_in=x)
        
        _, A, I = self.varToProp(var)
        
        A_fine = refineProperty_to_elemSize(A,torch.tensor([2.0]),self.frameFE.nodeXYbase,self.frameFE.connectivityBase,self.frameFE.elemSize,self.varSetup['varType'])
        I_fine = refineProperty_to_elemSize(I,torch.tensor([4.0]),self.frameFE.nodeXYbase,self.frameFE.connectivityBase,self.frameFE.elemSize,self.varSetup['varType'])

        
        # update the force vector to force_d - dummpy force at node of the output
        self.frameFE.applyForceOnNode(self.frameFE.bc['forces_d'])
        v,f_d = self.frameFE.solveFELin(self.matProp['E'].to(A_fine.dtype),A_fine,I_fine)
        
        # reset the force vector to original
        self.frameFE.applyForceOnNode(self.frameFE.bc['forces'])
        u,b = self.frameFE.solveFELin(self.matProp['E'].to(A_fine.dtype),A_fine,I_fine)

        freeDof = self.frameFE.freeDofs
        SE_v  = 0.5*(b[freeDof]*u[freeDof]).sum()
        MSE_v = (f_d[freeDof]*u[freeDof]).sum()
        # print("MSE_v, SE_v:", MSE_v.item(), SE_v.item())
               
        objValue_full = -1.0 * MSE_v / SE_v  # negative as we want to maximize MSE/SE
        
        objValue = objValue_full/torch.from_numpy(self.obj0)
    
        if Jac:
            # objValue.backward(retain_graph=True)    
            Sensitivity = torch.autograd.grad(objValue,x,retain_graph=True, create_graph=True)[0].detach().cpu().numpy()
        else:
            Sensitivity = 0*x.detach().cpu().numpy()
        
        objValue = objValue.detach().cpu().numpy()
        
        return objValue, Sensitivity
    
    
    
    def VolumeCon(self,x_in,con_value=0.5,Jac=True):
        x_in = x_in.reshape((-1,1))

        x = torch.tensor(x_in,requires_grad=True)
        
        var = self.apply_var_denormalize(var_in=x)
        
        _, A, _ = self.varToProp(var)
        Ax_fine = refineProperty_to_elemSize(A[:,0],torch.tensor([2.0]),self.frameFE.nodeXYbase,self.frameFE.connectivityBase,self.frameFE.elemSize,self.varSetup['varType'])
                        
        Volume = torch.einsum('i,i->i',self.frameFE.eleLength, Ax_fine.reshape(-1)).sum()
                
        v_con  = (Volume/torch.tensor([con_value]) - 1.0)**2
       
        if Jac:
            # v_con.backward(retain_graph=True)    
            Sensitivity = torch.autograd.grad(v_con,x,retain_graph=True, create_graph=True)[0].detach().cpu().numpy()
        else:
            Sensitivity = 0*x.detach().cpu().numpy()
        
        v_con = v_con.detach().cpu().numpy()
        
        return v_con.reshape((-1,1)), Sensitivity.reshape((len(v_con),len(x_in)))
        
    def ManufacturabilityCon(self,x_in, con_value=1.0,Jac=True):
        x = torch.tensor(x_in,requires_grad=True)
        
        var = self.apply_var_denormalize(var_in=x)
        
        numBeams = self.frameFE.connectivityBase.shape[0]
        numNodes = self.frameFE.nodeXYbase.shape[0]
       
        if self.varSetup['varType'] == 'Element':
            varShaped = var.clone().reshape((numBeams,-1))
        elif self.varSetup['varType'] == 'Node':
            varShaped = var.clone().reshape((numNodes,-1))
        elif self.varSetup['varType'] == 'Node+':
            varShaped = var.clone().reshape((numNodes + numBeams,-1))
        
        S_i, _, _ = self.varToProp(varShaped)

        m_con  = torch.tensor([con_value]) - S_i  # target value > 0.5 
        
        if Jac:
            
            jac = []
            for i in range(m_con.shape[0]):
                grad_i = torch.autograd.grad(
                    m_con[i],          # scalar output
                    x,                 # variable
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=False
                )[0].detach().cpu().numpy()                  # gradient vector shape (2,)
                jac.append(grad_i)
            
            Sensitivity = np.stack(jac, axis=0)   # shape (2,2)
                
        else:
            Sensitivity = np.zeros((len(m_con),len(x_in)))
        
        m_con = m_con.detach().cpu().numpy()
        
        return m_con.reshape((-1,1)), Sensitivity.reshape((len(m_con),len(x_in)))
                
    
    def dynamicCon(self, x_in, Jac=True):
        con_list = []
        grad_list = []
        
        for con_name, con_value in self.optimizationSetup['constraints'].items():
            
            # The function must be named e.g. VolumeCon, ManufacturabilityCon, etc.
            func_name = con_name + "Con"
            
            if not hasattr(self, func_name):
                raise ValueError(f"Missing constraint function: {func_name}")
            
            # Get the method dynamically
            func = getattr(self, func_name)
            
            # Call it
            c, c_grad = func(x_in, con_value, Jac=Jac)
            
            con_list.append(c)
            grad_list.append(c_grad)
        
        # Stack all constraints
        con = np.vstack(con_list)
        
        if Jac:
            con_grad = np.vstack(grad_list)
        else:
            con_grad = np.zeros((len(con), len(x_in)))
        
        return con.reshape((-1, 1)), con_grad.reshape((len(con), len(x_in)))

    def conNone(self,x_in,Jac=True):  
        con = -np.array([1.0])
       
        con_grad = np.zeros((1,len(x_in)))
        return con.reshape((-1,1)), con_grad.reshape((len(con),len(x_in)))
        
    def callbackFun(self,xk):
        self.iterations.append(len(self.iterations) + 1)
        # print("callBack fun xk =",xk)
        obj_v,_ = self.objectiveCall(xk)
        self.objective_values.append(obj_v)
        if self.conCall is not None:
            con_v,_ = self.conCall(xk)# type: ignore
            self.constraint_values.append(con_v.reshape(-1))
        
        # if self.iterations[-1]%5 ==0:
        #     self.plotIterVsObjAndCon(fig=plt.figure(2))
            
        # plt.close(2)
        # print(self.iterations)
        pass
        
    def plotIterVsObjAndCon(self,fig=None):
        if fig is None:
            fig, ax1 = plt.subplots()
        else:
            ax1 = fig.gca()
            plt.figure(fig.number)
        # --- Left axis: objective ---
        ax1.plot(self.iterations, self.objective_values, 'b-o', label='Objective')
        ax1.set_xlabel('Iteration',fontsize=16,fontweight='bold')
        ax1.set_ylabel('Objective Value', color='b',fontsize=16,fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='b')
    
        # --- Right axis: constraint ---
        ax2 = ax1.twinx()
        colors = ['red','green','orange','purple','pink','yellow','brown','cyan','black']
    
        # constraint_values shape: (N, m)
        for i in range(self.constraint_values[0].shape[0]):
            ax2.plot(self.iterations,
                     np.array(self.constraint_values)[:,i],
                     marker='s',
                     color=colors[i % len(colors)],
                     label=f'Constraint {i+1}')
    
        ax2.set_ylabel('Constraint Values', color='r',fontsize=16,fontweight='bold')
        ax2.tick_params(axis='y')
        ax2.set_ylim([-1.0,1.0])  # Set y-axis lower limit to 0 # type: ignore
    
        # Title
        plt.title(self.text + ' | Iteration vs Objective & Constraint')
    
        # Optional: combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    
        plt.tight_layout()
        plt.show()
        
    def optimizerRun(self,optimizationSetup,x0,algorithm='TNC',fileSaveLoc=None,options={'maxiter':100,'disp':True,'move_limit':0.5,'maxfun':500,'kkttol':1e-6,'miniter':20}):
        self.iterations = []
        self.objective_values = []
        self.constraint_values = []
        self.memo = {}
        
        self.fileSaveLoc = fileSaveLoc
        self.alpha = 0.0 # start with uniform phase 1 opt
        self.funCount = 0.0
        self.con = 0.0
        
        self.optimizationSetup = optimizationSetup
        self.ObjType = optimizationSetup['objective']
        
        xbest = x0.copy()
        obj_init,_= self.objectiveCall(xbest)
        # self.obj0 = obj_init.copy()
        
        fbest = 1.0
        if options['disp']:
            print("Normalized wrt = ", self.obj0)
        
        self.text = 'Optimization of objective '+self.ObjType + ' using ' + self.frameFE.AnalysisSettings['Type'] +' FEA and ' + algorithm +'\n'
        if options['disp']:
            print(self.text)
        
        self.conCall = None
        if self.optimizationSetup['constraints'] is not None:
            # conCall = self.volCon
            self.conCall = self.dynamicCon
        else:
            self.conCall = self.conNone
        
        
        self.callbackFun(xbest)# initial call to store initial obj and con values
        
        boundsWeights = Bounds([1e-4] * len(xbest), [1.] * len(xbest))  # type: ignore
        
        try :
            res = custom_minimize(self.objectiveCall,x0=xbest,constraintCall=self.conCall ,bounds=boundsWeights,\
                    method=algorithm,callback=self.callbackFun,\
                    options=options) # 8-i
            xbest = res.x.copy()
        except Exception as e:
            print("Optimization run failed with exception:", e)
            xbest = x0.copy()
                
        self.obj0 = self.obj0 * 0.0 + 1.0 # reset obj0;
        self.memo = {}
         
        obj_best,_ = self.objectiveCall(xbest) # dont take values from memory-new analysis

       
        # self.frameFE.plotStructure('',plotDeformed = True,TrueScale=True,fig=plt.figure(6),thicknessPlot = True,elemAnnotate=False,nodeAnnotate=False)
        # # plt.show(block=True)
        # plt.savefig(fileSaveLoc+'/DeformedRes_Phase'+str(i+1)+'.png')

        # self.frameFE.plotStructure('',plotDeformed = False,TrueScale=True,fig=plt.figure(7),thicknessPlot = True,elemAnnotate=False,nodeAnnotate=False)
        # plt.savefig(fileSaveLoc+'/unDeformedRes_Phase'+str(i+1)+ '.png')
    
        
        
        if options['disp'] or fileSaveLoc != None:
            self.plotIterVsObjAndCon(fig=plt.figure(2))
            if fileSaveLoc != None:
                plt.savefig(fileSaveLoc+'/IterVobj.png')
        
        # volume = self.getVolume(xbest)
        # print("Volume Opt = ", volume)
        return xbest.reshape(x0.shape), obj_best
    