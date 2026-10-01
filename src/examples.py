import numpy as np
import torch
# torch.set_default_dtype(torch.float64)
# torch.set_printoptions(precision=16)
# from mesher import arrowNumbering,boxXNumbering,arrowNumberingSymm,generateArrow3DMesh,generateArrow3DMeshTri,generateArrow3DMeshHexagon,generateVoronoi3DMeshHexagon
from mesher import MeshFrame

m = MeshFrame()
from boundaryCondition import boundaryConditionEx
def getExample(exampleCase,params = None):
    mf = MeshFrame()
    radiiNodIndex = None
    radiiElemIndex = None
    exampleNo = int(exampleCase[0])
    if(exampleNo == 1):
        exampleInfo ={'name':'One3DBeam','No':exampleCase}
        a=30.0
        nodeXY = np.array([0,0,0,a,0,0]).reshape(-1,3)
        connectivity = np.array([0,1]).astype(int).reshape(-1,2)
    if(exampleNo == 2):
        exampleInfo ={'name':'3DBeamBent','No':exampleCase}
        a=30.0
        nodeXY = np.array([0,0,0,0,0,a,a,0,a,a,a,a]).reshape(-1,3)
        connectivity = np.array([0,1,1,2,2,3]).astype(int).reshape(-1,2)
    
    
    if (exampleNo == 3): # mix of lattice
        exampleInfo ={'name':'DOV','No':exampleCase}
        if params == None:
            params = {'base': 2.,'height':5., 'theta': 0, 'alpha':0,'epsValue':0.,'phi':0.0,'delta':0, 'nx': 3, 'ny':3, 'nz':1,'r1':0.4,'r2':0.7,'Name':'DOV'}

        nodeXY,connectivity,radiiNodIndex,radiiElemIndex = mf.generateCombined3DLattice(params)
    
    
    if (exampleNo == 4 or exampleNo == 5 or exampleNo == 6): # mix of lattice
        exampleInfo ={'name':'DOV','No':exampleCase}
        if params == None:
            params = {'base': 2.,'height':5., 'theta': 0, 'alpha':0,'epsValue':0.,'phi':0.0,'delta':0, 'nx': 3, 'ny':3, 'nz':1,'r1':0.4,'r2':0.7,'Name':'DOV'}

        nodeXY,connectivity,radiiNodIndex,radiiElemIndex = mf.generateCombined3DLattice(params)
    
    
       
    bc = boundaryConditionEx(exampleCase,nodeXY,connectivity) # dynamic BC, changes with nodeXY based on prob
    
    if radiiNodIndex is None or radiiElemIndex is None:
        radiiElemIndex = np.arange(connectivity.shape[0]) # type: ignore
        radiiNodIndex = np.arange(nodeXY.shape[0]) # type: ignore

    return exampleInfo, nodeXY, connectivity, bc,radiiNodIndex,radiiElemIndex