import numpy as np

class MeshFrame():

    def refineFrameMesh(self,nodeXY, connectivity, numElemsPerBeam):
        numBeams = connectivity.shape[0]
        numnodeXY = nodeXY.shape[0]
        numNodesPerBeam = numElemsPerBeam+1;
        numNewNodesPerBeam = numElemsPerBeam-1
        numNodes = numnodeXY + numBeams*numNewNodesPerBeam;
        nodalCoords = np.zeros((numNodes,nodeXY.shape[1]));
        nodalCoords[0:numnodeXY,:] = nodeXY;
        conn = np.empty((0,2)); 
        t = np.linspace(0,1,numNodesPerBeam)
        for i in range (0,numBeams):
            
            startNode = nodalCoords[connectivity[i,0],:];
            endNode = nodalCoords[connectivity[i,1],:];
        
            xyz = np.outer((1-t),startNode) + np.outer(t,endNode);
            start = numnodeXY+numNewNodesPerBeam*(i)
            stop = numnodeXY+(i)*(numNewNodesPerBeam)+numNewNodesPerBeam
            newNodeNums = np.arange(start,stop);
            temp = np.repeat(newNodeNums,2);
            newConn = np.hstack((connectivity[i,0],temp,connectivity[i,1]));
            newConn = newConn.reshape((-1,2))
            nodalCoords[newNodeNums,:] = xyz[1:-1,:];
            conn = np.vstack((conn,newConn));
      
        return nodalCoords.astype(np.float64), conn.astype(int)
    
    def refineFrameMeshElemSize(self, nodeXY, connectivity, elemSize):
        numBeams = connectivity.shape[0]
        numnodeXY = nodeXY.shape[0]
        nodalCoords = np.copy(nodeXY)
        
        node1 = nodeXY[connectivity[:, 0]]
        node2 = nodeXY[connectivity[:, 1]]
        
        # Compute the Euclidean distance (L2 norm) between node1 and node2
        beamLength =  np.linalg.norm(node1 - node2, axis=1)                           
        
        numElemsPerBeam = np.rint(beamLength / np.array([elemSize])).astype(int)
        
        conn = np.zeros((np.sum(numElemsPerBeam),2))
        n = 0
        for i in range(numBeams):
            # Get start and end nodes for the beam
            startNode = nodalCoords[connectivity[i, 0], :]
            endNode = nodalCoords[connectivity[i, 1], :]
            
            # Determine the number of elements for this beam
            numNodesPerBeam = numElemsPerBeam[i] + 1
            numNewNodesPerBeam = numElemsPerBeam[i] - 1
            
            # Create a parameterized line for nodes along the beam
            t = np.linspace(0, 1, numNodesPerBeam)
            xyz = np.outer(1 - t, startNode) + np.outer(t, endNode)
            
            # Add new nodes to nodalCoords
            newNodeNums = np.arange(numnodeXY, numnodeXY + numNewNodesPerBeam)
            nodalCoords = np.vstack((nodalCoords, xyz[1:-1, :]))
            numnodeXY += numNewNodesPerBeam  # Update the total number of nodes
            
            temp = np.repeat(newNodeNums,2);
            newConn = np.hstack((connectivity[i,0],temp,connectivity[i,1]));
            newConn = newConn.reshape((-1,2))
            conn[n:n+newConn.shape[0],:] = newConn
            n += newConn.shape[0]
    
        return nodalCoords.astype(np.float64), conn.astype(int)
    
    def remeshFrame(self,nodeXY,connMat,elemsToBeDeleted):
    
        # Some of this should be available in self
    
        numNodes = np.max(connMat)
    
        connMat = np.delete(connMat, elemsToBeDeleted, 0)
        
        # next check for missing nodes if any
        
        nodeXYUnique = np.unique(connMat)
    
        reference_arr = np.arange(numNodes + 1)
    
        missingNodes =  np.setdiff1d(reference_arr, nodeXYUnique)
    
        nodeXY = np.delete(nodeXY,missingNodes,0)
        
        missingNodes = np.flip(missingNodes)

        for i in range(missingNodes.shape[0]):
    
            mask = connMat > missingNodes[i]
    
            connMat[mask] -= 1
    
        return nodeXY, connMat
  
    def boxXNumbering(self,params=None):
        if params is None:
            params = {'base': 1, 'height': 1, 'nCols': 3, 'nRows': 4}
            
        numVertices = 2*(params['nRows'])*(params['nCols']) + params['nRows']+params['nCols'] +1
        
        vertices = np.zeros((numVertices, 2))
        
        xCo = np.arange(params['nCols']+1)*params['base']
        
        yCo = np.arange(params['nRows']+1)*params['height']
        
        Xcorner,Ycorner = np.meshgrid(xCo,yCo)
        
        xCo = (np.arange(1,params['nCols']+1)-0.5)*params['base']
        
        yCo = (np.arange(1,params['nRows']+1)-0.5)*params['height']
        
        Xmid,Ymid = np.meshgrid(xCo,yCo)
        
        vertices[:,0] = np.concatenate((Xcorner.reshape(-1), Xmid.reshape(-1)),axis=None)
        vertices[:,1] = np.concatenate((Ycorner.reshape(-1), Ymid.reshape(-1)),axis=None)
        
        nBeams = params['nCols']*(params['nRows']+1) + params['nRows']*(params['nCols']+1) + 4*params['nRows']*params['nCols']
        connMat = np.zeros((nBeams,2))
        k=0
        # Element forming in x direction
        for j in range(params['nRows']+1):
            for i in range(params['nCols']):
                connMat[k,:] = np.array([i+j*(params['nCols']+1),(i+1)+j*(params['nCols']+1)])
                k=k+1
        # Element forming in y direction
        for j in range(params['nRows']):
            for i in range(params['nCols']+1):
                connMat[k,:] = np.array([i+j*(params['nCols']+1),i+(j+1)*(params['nCols']+1)])
                # print(connMat[k,:])
                k=k+1
                
        # Elment forming in ceter of each box (4 elements)
        e = (params['nRows']+1)*(params['nCols']+1)     
        for j in range(params['nRows']):
            for i in range(params['nCols']):
                a,b=i+j*(params['nCols']+1),(i+1)+j*(params['nCols']+1)
                c,d = i+(j+1)*(params['nCols']+1), (i+1)+(j+1)*(params['nCols']+1)
                # print(e)
                connMat[k,:] = np.array([a,e])
                k=k+1;
                connMat[k,:] = np.array([b,e])
                k=k+1;
                connMat[k,:] = np.array([c,e])
                k=k+1;
                connMat[k,:] = np.array([d,e])
                k=k+1;
                e = e+1
                
        return vertices,connMat.astype(int)
    
    
    def arrowNumbering(self,params=None):
        if params is None:
            params = {'base': 2, 'theta': 60, 'alpha': 15, 'nRows': 2, 'nCols': 3}
        numVertices = (params['nRows']+1)*(params['nCols']*2+1) - 2
        numVerticesPerRow = (params['nCols']*2+1)
        numVerticesinTopRow = (params['nCols']*2-1)
    
        nodeNumbers = np.arange(1,numVertices+1)
        ind2remove = numVerticesinTopRow+1 # index to insert
        nodeNumbers = np.insert(nodeNumbers,0,[0])
    
        nodeNumbers = np.insert(nodeNumbers, ind2remove, [0]) # insert
        nodeNumbers = nodeNumbers.reshape((params['nRows']+1, params['nCols']*2+1))
        
        nArrowHeads = params['nRows']*params['nCols']+params['nRows']*(params['nCols']-1)
        nUpwardArrowHeads = params['nRows']*params['nCols']
        nDownwardArrowHeads = nArrowHeads - params['nRows']*params['nCols']
        nElems = nUpwardArrowHeads*4 + 2*(params['nCols']-1)
        connMat = np.repeat(nodeNumbers, 2).reshape([nodeNumbers.shape[0],-1])
        connMat = connMat[:,1:-1].reshape((-1,2))  
        connMat = np.delete(connMat, ind2remove-1, axis=0)  
        connMat = np.delete(connMat, 0, axis=0)  
    
       
        k = connMat.shape[0]
        for i in range(params['nRows']):
            for j in range(0, params['nCols']*2, 2):
                k += 1
                connMat = np.vstack([connMat, [nodeNumbers[i+1,j], nodeNumbers[i,j+1]]])
                k += 1
                connMat = np.vstack([connMat, [nodeNumbers[i,j+1], nodeNumbers[i+1,j+2]]])
    
    
        B = params['base']
        b = 0.5*params['base']
        H = b * np.tan(params['theta']*np.pi/180)
        h = b * np.tan(params['alpha']*np.pi/180)
        vertices = np.zeros((numVerticesPerRow, 2))
        for i in range(1, params['nCols']+1):
            vertices[2*i+1-1,0] = i*B
            vertices[2*i-1,0] = i*B-b
            vertices[2*i-1,1] = h
    
        # print(vertices)
        # import sys
        # sys.exit()
        addHeight = H-h
        refVertices = vertices
        for i in range(1, params['nRows']+1):
            vertices = np.vstack([refVertices+[0, i*addHeight], vertices])
    
        vertices = np.delete(vertices, ind2remove, axis=0)
        vertices = np.delete(vertices, 0, axis=0)
    
        connMat = np.sort(connMat, axis=1)
        return vertices,connMat-[1,1]
    
    
    def arrowNumberingSymm(self,params=None):
        
        vertices,connMat = arrowNumbering(params) # type: ignore
        
        top = np.where(vertices[:,1] == np.max(vertices[:,1]))
        
        b = 0.5*params['base'] # type: ignore
        H = b * np.tan(params['theta']*np.pi/180) # type: ignore
        h = b * np.tan(params['alpha']*np.pi/180) # type: ignore
            
        y1 = vertices[top[0][0],1]-h
        
        x1 = np.min(vertices[:,0])
        x2 = np.max(vertices[:,0])
        
        arr = np.array([[x1,y1],[x2,y1]])
        vertices = np.concatenate((vertices,arr))
        
        conn = np.array([[top[0][0],np.max(connMat)+1],[top[0][-1],np.max(connMat)+2]])
        
        connMat = np.concatenate((connMat,conn))
        
        return vertices,connMat
        
    def arrow3Dunit(self,params=None):
      if params is None:
        params = {'base': 5., 'theta': 80, 'alpha': 15,'epsValue':0.0} 
      numVertices = 10
      vertices = np.zeros((numVertices, 3),dtype=np.float64)
    
      nodeNumbers = np.arange(0,numVertices)
      B = params['base']
      b = 0.5*params['base']
      H = b * np.sqrt(2) * np.tan(params['theta']*np.pi/180)
      h = b * np.sqrt(2) * np.tan(params['alpha']*np.pi/180)
      vertices[0] = [0,0,0]
      vertices[1] = [B,0,0]
      vertices[2] = [B,B,0]
      vertices[3] = [0,B,0]
      vertices[4] = [b,b,h]     
      vertices[5] = [b,b,H]
      
      epsValue =params['epsValue'];
      M7 = (vertices[0,:]+ vertices[5,:])/2;
      M8 = (vertices[1,:]+ vertices[5,:])/2;
      M9 = (vertices[2,:]+ vertices[5,:])/2;
      M10 = (vertices[3,:]+ vertices[5,:])/2;
    
      M7[0:2] += np.array([-1,-1])*epsValue;
      M8[0:2] += np.array([1,-1])*epsValue;
      M9[0:2] +=  np.array([1,1])*epsValue;
      M10[0:2] += np.array([-1,1])*epsValue;
      
      vertices[6] = M7;vertices[7] = M8;vertices[8] = M9;vertices[9] = M10;
      
      connMat = np.array([[3,4],\
                          [0,4],\
                          [1,4],\
                          [2,4],\
                          [0,6],\
                          [1,7],\
                          [2,8],\
                          [3,9],\
                          [5,6],\
                          [5,7],\
                          [5,8],\
                          [5,9]],dtype=int)
      
      return vertices, connMat
    
    def kelvin3Dunit(self,params=None):
      if params is None:
        params = {'base': 5.,'height':5., 'theta': 0, 'alpha': 0,'epsValue':0} 
      
      def rotCord(theta, axis='x'):
        """
        This function takes an angle of rotation theta in radians and an optional axis of rotation, 
        and returns the corresponding rotation matrix.
      
        Args:
            theta: The angle of rotation in radians.
            axis: The axis of rotation. Defaults to 'x'.
      
        Returns:
            A rotation matrix.
        """
        if axis == 'x':
          return [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
        elif axis == 'y':
          return [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
        elif axis == 'z':
          return [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        else:
          raise ValueError("Invalid axis")
          
      numVertices = 24
      vertices = np.zeros((numVertices, 3),dtype=np.float64)
        
      nodeNumbers = np.arange(0,numVertices)
      B = params['base']
      H = params['height']
      
      vertices[0] = [-B/np.sqrt(2),0,-H/2]
      vertices[1] = [0,-B/np.sqrt(2),-H/2]
      vertices[2] = [ B/np.sqrt(2),0,-H/2]
      vertices[3] = [0, B/np.sqrt(2),-H/2]
      
      
      vertices[4:8,:] = (rotCord(np.pi/2, axis='y')@vertices[0:4,:].T).T
      vertices[8:12,:] = (rotCord(np.pi, axis='y')@vertices[0:4,:].T).T
      vertices[12:16,:] = (rotCord(-np.pi/2, axis='y')@vertices[0:4,:].T).T
      
      vertices[16:20,:] = (rotCord(np.pi/2, axis='x')@vertices[0:4,:].T).T
      vertices[20::,:] = (rotCord(-np.pi/2, axis='x')@vertices[0:4,:].T).T
      
      vertices = vertices - np.min(vertices,axis=0)
      
      connMat = np.array([[0,1],\
                          [1,2],\
                          [2,3],\
                          [3,0]],dtype=int)
      connMat2 = np.array([[0,6],\
                          [1,23],\
                          [2,12],\
                          [3,17],\
                          [5,20],\
                          [22,13],\
                          [15,18],\
                          [16,7],\
                          [10,4],\
                          [9,21],\
                          [8,14],\
                          [11,19],\
                          ],dtype=int)
      connMat = np.concatenate((connMat,connMat+4,connMat+8,connMat+12,connMat+16,connMat+20,connMat2),dtype=int)
    
      
      return vertices, connMat
    
    def octet3Dunit(self,params=None):
      if params is None:
        params = {'base': 5.,'height':5., 'theta': 0, 'alpha': 0,'epsValue':0} 
      vertices = np.array([0, 0,	0,
            1,	0,	0,
            1,	1,	0,
            0,	1,	0,
            0.5,	0.5,	0,
            0.5,	0,	0.5,
            1, 	0.5,	0.5,
            0.5,	1,	0.5,
            0,	0.5,	0.5,
            0,	0,	1,
            1,	0,	1,
            1,	1,	1,
            0,	1,	1,
            0.5,	0.5,	1]).reshape(-1,3);
      vertices[:,0:2] = vertices[:,0:2]*params['base']
      vertices[:,2] = vertices[:,2]*params['height']
      connMat = np.array([0,	4,
            1,	4,
            2,	4,
            3,	4,
            # 5,	6,
            # 6,	7,
            # 7,	8,
            # 8,	5,
            9,	13,
            10,	13,
            11,	13,
            12,	13,
            0,	8,
            9,	8,
            12,8,
            3,	8,
            0,	5,
            1,	5,
            10,	5,
            9,	5,
            1,	6,
            2,	6,
            11,	6,
            10,	6,
            2,	7,
            3,	7,
            12,	7,
            11,	7,
            5,	4,
            5,	13,
            6,	4,
            6,	13,
            7,	4,
            7,	13,
            8,	4,
            8,	13],dtype=int).reshape(-1,2);
      
      return vertices, connMat
    
    def delaunay3Dunit(self,params=None):
      if params is None:
        params = {'base': 5.,'height':5., 'theta': 0, 'alpha': 0,'epsValue':0} 
      
      vertices = np.array([0, 0,	0,
            1,	0,	0,
            1,	1,	0,
            0,	1,	0,
            0,	0,	1,
            1,	0,	1,
            1,	1,	1,
            0,	1,	1,
            0.5, 0.5, 0.5,
            0.5,	0,	0.5,
            1, 	0.5,	0.5,
            0.5,	1,	0.5,
            0,	0.5,	0.5,
            0.5,	0.5,	0,
            0.5,	0.5,	1]).reshape(-1,3);
      vertices[:,0:2] = vertices[:,0:2]*params['base']
      vertices[:,2] = vertices[:,2]*params['height']
      
      connMat = np.array([0,1,1,2,2,3,3,0,
      4,5,5,6,6,7,7,4,
      0,4,1,5,2,6,3,7,
      8,0,8,1,8,2,8,3,8,4,8,5,8,6,8,7,
      8,9,8,10,8,11,8,12,8,13,8,14],dtype=int).reshape(-1,2);
            
      return vertices, connMat
    
    def SCBCC3Dunit(self,params=None):
      if params is None:
        params = {'base': 5.,'height':5., 'theta': 0, 'alpha': 0,'epsValue':0} 
      
      vertices = np.array([0, 0,	0,
            1,	0,	0,
            1,	1,	0,
            0,	1,	0,
            0,	0,	1,
            1,	0,	1,
            1,	1,	1,
            0,	1,	1,
            0.5, 0.5, 0.5]).reshape(-1,3);
      vertices[:,0:2] = vertices[:,0:2]*params['base']
      vertices[:,2] = vertices[:,2]*params['height']
      
      connMat = np.array([0,1,1,2,2,3,3,0,
      4,5,5,6,6,7,7,4,
      0,4,1,5,2,6,3,7,
      8,0,8,1,8,2,8,3,8,4,8,5,8,6,8,7],dtype=int).reshape(-1,2);
            
      return vertices, connMat
    
    def rhombic3Dunit(self,params=None):
        if params is None:
            params = {'base': 5.,'height':5., 'theta': 0, 'alpha': 0,'epsValue':0} 
              
        vertices = np.array([
        # 0, 0,	0, 
        # 1,	0,	0,
        # 1,	1,	0,
        # 0,	1,	0,
        # 0,	0,	1,
        # 1,	0,	1,
        # 1,	1,	1,
        # 0,	1,	1,
        0.5, 0, 0.5,
        1, 	0.5,	0.5,
        0.5,	1,	0.5,
        0,	0.5,	0.5,
        0.5,	0.5,	0, #bot center
        0.5,	0.5,	1., #top
        0.25, 0.25, 0.25,
        0.75, 0.25, 0.25,
        0.75, 0.75, 0.25,
        0.25, 0.75, 0.25,
        0.25, 0.25, 0.75,
        0.75, 0.25, 0.75,
        0.75, 0.75, 0.75,
        0.25, 0.75, 0.75]).reshape(-1,3);
        
        vertices[:,0:2] = vertices[:,0:2]*params['base']
        vertices[:,2] = vertices[:,2]*params['height']
        
        # connMat = np.array([0,14,1,15,2,16,3,17,4,18,5,19,6,20,7,21,
        # 14,8,14,11,15,8,15,9,16,9,16,10,17,10,17,11,
        # 18,8,18,11,19,8,19,9,20,9,20,10,21,10,21,11,
        # 12,14,12,15,12,16,12,17,13,18,13,19,13,20,13,21],dtype=int).reshape(-1,2);
        
        #removed few members
        connMat = np.array([14,8,14,11,15,8,15,9,16,9,16,10,17,10,17,11,
        18,8,18,11,19,8,19,9,20,9,20,10,21,10,21,11,
        12,14,12,15,12,16,12,17,13,18,13,19,13,20,13,21],dtype=int).reshape(-1,2)-8;
        
        return vertices, connMat
    def cuboctahedron3Dunit(self,params=None,n_sides=8):
      if params is None:
        params = {'base': 5.,'height':5., 'theta': 0, 'alpha': 0,'epsValue':0} 
      
      def rotCord(theta, axis='x'):
        """
        This function takes an angle of rotation theta in radians and an optional axis of rotation, 
        and returns the corresponding rotation matrix.
      
        Args:
            theta: The angle of rotation in radians.
            axis: The axis of rotation. Defaults to 'x'.
      
        Returns:
            A rotation matrix.
        """
        if axis == 'x':
          return [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
        elif axis == 'y':
          return [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
        elif axis == 'z':
          return [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        else:
          raise ValueError("Invalid axis")
          
      
      B = params['base']
      H = params['height']
      
    #   n_sides = 6
    #   print(n_sides)
      beta =1.;
      if n_sides== 4 or n_sides == 6 or n_sides == 10:
        beta = 0. #skip rotate by half angle
      # Calculate the radius R using the generalized formula
    #   R = B / (2 * np.sin(np.pi / n_sides))
      R = B/np.sqrt(2)
  
      # Angle between the vertices
      angle = np.linspace(0, 2 * np.pi, n_sides + 1)
      angle = angle - beta*angle[1]/2 # rotate the octagon by half the distance
  
      # X and Y coordinates of the polygon
      x = R * np.cos(angle[0:-1])
      y = R * np.sin(angle[0:-1])
      
      numVertices = int(n_sides*6)
      vertices = np.zeros((numVertices, 3),dtype=np.float64)
        
      nodeNumbers = np.arange(0,numVertices)
      
      vertices[0:n_sides,0] = x
      vertices[0:n_sides,1] = y
      vertices[0:n_sides,2] = -H/2
      
      vertices[n_sides:int(n_sides*2),:] = (rotCord(np.pi, axis='y')@vertices[0:n_sides,:].T).T
      vertices[int(n_sides*2):int(n_sides*3),:] = (rotCord(np.pi/2, axis='y')@vertices[0:n_sides,:].T).T
      vertices[int(n_sides*3):int(n_sides*4),:] = (rotCord(-np.pi/2, axis='y')@vertices[0:n_sides,:].T).T
      
      vertices[int(n_sides*4):int(n_sides*5),:] = (rotCord(np.pi/2, axis='x')@vertices[0:n_sides,:].T).T
      vertices[int(n_sides*5)::,:] = (rotCord(-np.pi/2, axis='x')@vertices[0:n_sides,:].T).T
      
      vertices = vertices - np.min(vertices,axis=0)
      
      connMatbase = (np.concatenate((np.arange(n_sides),np.arange(1,n_sides+1))).reshape((-1,n_sides)).T)

      connMatbase[connMatbase == n_sides] = 0
      
      connMat = connMatbase
      # connectivity generation of each of 6 polygons
      for i in range(5):
          connMat = np.concatenate((connMat,connMatbase+n_sides*(i+1)),dtype=int)
      
      # lets find nodes closest to top and bot panels
      # helps find the connections between the polygons
      def find_close_pairs(vertices,arr1,arr2):
        one_octagon = vertices[arr1, :]
        rest_octagons = vertices[arr2, :]
        distances = np.linalg.norm((one_octagon - rest_octagons[:,np.newaxis,:]),axis=2)
        min_distances = np.round(np.min(distances, axis=0),decimals=4)
        min_row_indices = np.argmin(distances, axis=0)
        # print(min_row_indices)
        has_repeats = len(min_row_indices) != len(np.unique(min_row_indices))
        if has_repeats:
          v = np.where(min_distances == np.min(min_distances))[0]
        else:
          v = np.arange(0,len(min_row_indices))
        # print(min_distances)
        # print(v)
        conn = np.concatenate((arr1[v],min_row_indices[v]+arr2[0])).reshape((-1,arr1[v].shape[0])).T
        return conn
      
      for i in range(4): # top, bot, left and right side connections
        conn = find_close_pairs(vertices,np.arange(int(n_sides*i),int(n_sides*(i+1))),np.arange(int(n_sides*(i+1)),int(n_sides*6)))
        connMat = np.concatenate((connMat,conn),dtype=int)      
            
      return vertices, connMat
      
    def arrow3DunitTri(self,params=None):
      if params is None:
        params = {'base': 5., 'theta': 80, 'alpha': 15} 
      numVertices = 5
      vertices = np.zeros((numVertices, 3),dtype=np.float64) 
    
      nodeNumbers = np.arange(0,numVertices)
      B = params['base']
      b = 0.5*params['base']
      H = b * np.tan(params['theta']*np.pi/180)
      h = b * np.tan(params['alpha']*np.pi/180)
      vertices[0] = [0,0,0]
      vertices[1] = [B,0,0]
      vertices[2] = [b,b*np.sqrt(3),0]
      vertices[3] = [b,b/np.sqrt(3),h]
      vertices[4] = [b,b/np.sqrt(3),H]
      
      connMat = np.array([[0,3],\
                          [1,3],\
                          [2,3],\
                          [0,4],\
                          [1,4],\
                          [2,4]],dtype=int)
    
      return vertices, connMat
    
    # def generateArrow3DMesh(self,params):
    #   if params is None:
    #     params = {'base': 2, 'theta': 60, 'alpha': 30, 'nx': 2, 'ny': 2, 'nz': 2} 
    
    #   vertices, connMat = arrow3Dunit(params) # generates a unit cell which we repeat
    #   nx = params['nx'] # Number of repetitions along the X axis
    #   ny = params['ny']  # Number of repetitions along the Y axis
    #   nz = params['nz']  # Number of repetitions along the Z axis
    
    #   unique_vertices_dict = {}
    
    #   # Lists to store the indices of unique vertices and connectivity matrices
    #   unique_vertex_indices = []
    #   adjusted_connMat_list = []
    
    #   for k in range(nz):
    #       for j in range(ny):
    #           for i in range(nx):
    #               # Translate the vertices of the unit cell
    #               translated = vertices.copy()
    #               translated[:, 0] += i * vertices[1, 0]  # Translate along the X axis
    #               translated[:, 1] += j * vertices[2, 1]  # Translate along the Y axis
    #               translated[:, 2] += k * (vertices[5, 2] - vertices[4,2])  # Translate along the Z axis and match the points
    
    #               # Add unique vertices to the dictionary, keeping the smallest values
    #               for vertex in translated:
    #                   key = tuple(vertex)
    #                   if k==2:
    #                       print(key not in unique_vertices_dict)
    #                       print(key in unique_vertices_dict)
    #                   if key not in unique_vertices_dict:
    #                       unique_vertex_indices.append(len(unique_vertices_dict))
    #                       unique_vertices_dict[key] = vertex
    #                   else:
    #                       unique_vertex_indices.append(list(unique_vertices_dict.keys()).index(key))
    
    
    
    #               # Adjust the connectivity matrix for the current repetition using unique vertex indices
    #               adjusted_connMat = []
    #               for conn in connMat:
    #                   conn_indices = [unique_vertex_indices[conn[0] + (i + j * nx + k * nx * ny) * len(vertices)],
    #                                   unique_vertex_indices[conn[1] + (i + j * nx + k * nx * ny) * len(vertices)]]
    #                   adjusted_connMat.append(conn_indices)
    #               adjusted_connMat_list.extend(adjusted_connMat)
    
    
    #   # Convert the dictionary values to a NumPy array
    #   all_vertices = np.array(list(unique_vertices_dict.values()))
    
    #   # Convert the adjusted connectivity matrix to a NumPy array
    #   all_connMat = np.array(adjusted_connMat_list)
      
    #   return all_vertices, all_connMat
    
    def combined3Dunit(self,params=None):
        if params is None:
          params = {'base': 5.,'height':5., 'theta': 0, 'alpha': 0,'epsValue':0,'Name':'DOV'} 
        vertices_list = []
        connMat_list = []
        # for i in range(len(params['Name'])): #
        
        if params['Name']=='D':
            vertices,connMat = self.delaunay3Dunit(params)
        elif params['Name']=='O': 
            vertices,connMat = self.octet3Dunit(params)
        elif params['Name']=='K':# kelvin or vornoi lattice
            params['base'] = params['base']/2
            vertices,connMat = self.kelvin3Dunit(params)
            params['base'] = params['base']*2
        elif params['Name'] == 'R': # robhic
            vertices,connMat = self.rhombic3Dunit(params)
        elif params['Name'][0] == 'C':
            params['base'] = params['base']/2
            vertices,connMat = self.cuboctahedron3Dunit(params,np.array(params['Name'][1:]).astype(int)) # type: ignore
            params['base'] = params['base']*2
        elif params['Name'] == 'A':
            vertices,connMat = self.arrow3Dunit(params)      
        elif params['Name']=='V': # v for vector lattice
            params['base'] = params['base']/np.sqrt(2)
            vertices,connMat = self.kelvin3Dunit(params)
            params['base'] = params['base']*np.sqrt(2)
        elif params['Name']=='B': # body centered cubic
            vertices,connMat = self.SCBCC3Dunit(params)
            
        vertices_list.append(vertices)
        connMat_list.append(connMat)
        
        all_vertices, all_connMat = self.merge_vertices_and_conn(vertices_list, connMat_list)
        # print(all_connMat)
        all_vertices, all_connMat = self.find_and_divide_intersections(all_vertices, all_connMat)
      
        # all_vertices, all_connMat = self.refineFrameMesh(all_vertices, all_connMat, 1)
        return all_vertices, all_connMat
    
    def generateCombined3DLattice(self,params): # any shape in either hexagon or square
        if params is None:
            params = {'base': 2.,'height':5., 'theta': 0, 'alpha':0,'epsValue':0.,'phi':0.0,'delta':0, 'nx': 3, 'ny':3, 'nz':1,'Name':'V','Shape':'Square'}
    
        vertices, connMat = self.combined3Dunit(params) # generates a unit cell which we repeat
        nx = params['nx']  # Number of repetitions along the X axis
        ny = params['ny']  # Number of repetitions along the Y axis
        nz = params['nz']  # Number of repetitions along the Z axis
    
        all_vertices = vertices.copy()
        all_connMat = connMat.copy()
        all_radii_indices = np.arange(vertices.shape[0])  # Start with the indices of the nodal radii of the unit cell
        temp_radii_indices = all_radii_indices.copy()  # Indices for nodal radii from the unit cell
    
        unit_cell_size = connMat.shape[0]  # Store the size of the unit cell's connectivity matrix
        Mov1 = vertices.copy()
        Mov1[:, [0, 1]] += -params['base'] / 2
        
        # Index array that maps each connection in the full lattice to the corresponding connection in the unit cell
        conn_index_array = np.arange(unit_cell_size)  # Initial index array for the first unit cell
        all_conn_indices = conn_index_array.copy()
    
        for k in range(nz):
            for j in range(ny):
                if params['Shape'] == 'Hexagon':  # Add extra lattice unit per side
                    if (j < 1) or j >= ny - 1:
                        etr = 0
                    elif j < 3 or j >= ny - 3:
                        etr = 1  # This should be 1 for panther project, 0 for other reasons
                    elif j < 5 or j >= ny - 5:
                        etr = 2
                    else:
                        etr = 3
                else:
                    etr = 0
    
                for i in range(nx + int(2 * etr)):
                    angRad = np.deg2rad(params['phi'] * (i + j))
                    c = np.cos(angRad)
                    s = np.sin(angRad)
                    RotM = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
                    Mov2 = (RotM @ Mov1.T).T
                    Mov2[:, [0, 1]] += params['base'] / 2
    
                    translated = Mov2
                    translated[:, 0] += i * (params['base'] + params['delta']) - min(j, etr) * params['base']
                    translated[:, 1] += j * (params['base'] + params['delta'])
                    translated[:, 2] += k * (params['height'])
    
                    if i + j + k != 0:
                        connMat_temp = -connMat
                        current_indices = conn_index_array.copy()  # Create a copy of the index array for this repetition
                        
                        for vertex, node_num in zip(translated, range(translated.shape[0])):
                            vertice_loc = np.argwhere((np.sum(np.isclose(vertex, all_vertices), axis=1) == all_vertices.shape[1]) == 1)
                            vertice_size = vertice_loc.size
                            connMat_loc = np.argwhere(connMat_temp == -node_num)
    
                            if vertice_size == 0:  # New vertex
                                all_vertices = np.append(all_vertices, vertex.reshape((1, -1)), axis=0)
                                all_radii_indices = np.append(all_radii_indices, temp_radii_indices[node_num])  # Append the corresponding radius index
                                connMat_temp[connMat_loc[:, 0], connMat_loc[:, 1]] = all_vertices.shape[0] - 1
                            else:  # Repeated vertex
                                merged_node = vertice_loc[0, 0]
                                connMat_temp[connMat_loc[:, 0], connMat_loc[:, 1]] = merged_node
                                
                                # No need to average radii indices; just keep the first occurrence
    
                        sumTemp = np.sum(connMat_temp, axis=1)
                        allSum = np.sum(all_connMat, axis=1)
    
                        checkPoints = np.where(1.0 * (sumTemp == allSum.reshape(-1, 1)) == 1.0)
                        removeElem = []
    
                        for cp in range(len(checkPoints[0])):
                            cond1 = connMat_temp[checkPoints[1][cp], :] == all_connMat[checkPoints[0][cp], :]
                            cond2 = connMat_temp[checkPoints[1][cp], :] == np.flip(all_connMat[checkPoints[0][cp], :])
    
                            if np.sum(cond1) == 2 or np.sum(cond2) == 2:
                                removeElem.append(cp)
                        
                        connMat_temp = np.delete(connMat_temp, checkPoints[1][removeElem], axis=0)
                        current_indices = np.delete(current_indices, checkPoints[1][removeElem], axis=0)
                        
                        all_connMat = np.concatenate((all_connMat, connMat_temp))
                        all_conn_indices = np.concatenate((all_conn_indices, current_indices))  # Append the current index array
    
        all_vertices = all_vertices - np.min(all_vertices, axis=0)
    
        return all_vertices, all_connMat, all_radii_indices, all_conn_indices
        
    def generateArrow3DMesh(self,params):
        if params is None:
            params = {'base': 5., 'theta': 80, 'alpha':15,'epsValue':0.,'phi':0.0,'delta':2.5, 'nx': 1, 'ny': 1, 'nz':2,'r1':0.4,'r2':0.7}
    
      
        vertices, connMat = arrow3Dunit(params) # generates a unit cell which we repeat # type: ignore
        nx = params['nx'] # Number of repetitions along the X axis
        ny = params['ny']  # Number of repetitions along the Y axis
        nz = params['nz']  # Number of repetitions along the Z axis
      
        unique_vertices_dict = {}
      
        # Lists to store the indices of unique vertices and connectivity matrices
        unique_vertex_indices = []
        adjusted_connMat_list = []
    
        all_vertices = vertices
        all_connMat = []
        #move vertices to origin
        Mov1 = vertices.copy()
        Mov1[:,[0,1]] += -params['base']/2;
                    
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # Translate the vertices of the unit cell
                    angRad = np.deg2rad(params['phi']*(i+j))
                    c = np.cos(angRad)
                    s = np.sin(angRad)
                    RotM = np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                    
                    #rotate about origin
                    Mov2 = (RotM@Mov1.T).T
                    
                    #move the vertices back to location
                    Mov2[:,[0,1]] += params['base']/2;
                    
                    translated = Mov2;
                    translated[:, 0] += i * (vertices[1, 0] + params['delta'])  # Translate along the X axis
                    translated[:, 1] += j * (vertices[2, 1] + params['delta'])  # Translate along the Y axis
                    translated[:, 2] += k * (vertices[5, 2] - vertices[4,2])    # Translate along the Z axis and match the points
            
                    connMat_temp = -connMat
                    # Add unique vertices to the dictionary, keeping the smallest values
                    for vertex, node_num in zip(translated,range(translated.shape[0])):
                        vertice_loc = np.argwhere((np.sum(np.isclose(vertex,all_vertices),axis=1)==all_vertices.shape[1])==1)
                        vertice_size = vertice_loc.size
                        connMat_loc = np.argwhere(connMat_temp==-node_num)
                        if vertice_size == 0: 
                            all_vertices = np.append(all_vertices,vertex.reshape((1,-1)),axis=0)
                            connMat_temp[connMat_loc[:,0],connMat_loc[:,1]]=all_vertices.shape[0]-1
                        else:
                            # print(connMat[connMat_loc[0,:],connMat_loc[1,:]])
                            connMat_temp[connMat_loc[:,0],connMat_loc[:,1]]=vertice_loc
                            
                    all_connMat.append(connMat_temp)
        
        # # Convert the dictionary values to a NumPy array
        # all_vertices = np.array(list(unique_vertices_dict.values()))
        
        # # Convert the adjusted connectivity matrix to a NumPy array
        all_connMat = np.array(all_connMat).reshape((-1,2))
          
        return all_vertices, all_connMat
        
    def generateArrow3DMeshHexagon(self,params): # arrowhead inside a hexagon
        if params is None:
            params = {'base': 5., 'theta': 80, 'alpha':15,'epsValue':0.,'phi':0.0,'delta':2.5, 'nx': 1, 'ny': 1, 'nz':2,'r1':0.4,'r2':0.7}
      
        vertices, connMat = arrow3Dunit(params) # generates a unit cell which we repeat # type: ignore
        nx = params['nx'] # Number of repetitions along the X axis
        ny = params['ny']  # Number of repetitions along the Y axis
        nz = params['nz']  # Number of repetitions along the Z axis
      
        unique_vertices_dict = {}
      
        # Lists to store the indices of unique vertices and connectivity matrices
        unique_vertex_indices = []
        adjusted_connMat_list = []
    
        all_vertices = vertices
        all_connMat = []
        #move vertices to origin
        Mov1 = vertices.copy()
        Mov1[:,[0,1]] += -params['base']/2;
                    
        for k in range(nz):
            for j in range(ny):
                # add extra arrowheads per side     
                if (j<1) or j>=ny-1:
                    etr = 0;
                elif j<3 or j>= ny-3:
                    etr = 1;
                elif j<5 or j>= ny-5:
                    etr = 2;
                else:
                    etr = 3;
                    
                for i in range(nx+2*etr):
                    # Translate the vertices of the unit cell
                    angRad = np.deg2rad(params['phi']*(i+j))
                    c = np.cos(angRad)
                    s = np.sin(angRad)
                    RotM = np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                    
                    #rotate about origin
                    Mov2 = (RotM@Mov1.T).T
                    
                    #move the vertices back to location
                    Mov2[:,[0,1]] += params['base']/2;
                    
                    translated = Mov2;
                    translated[:, 0] += i * (vertices[1, 0] + params['delta']) - min(j,etr)*params['base']  # Translate along the X axis
                    translated[:, 1] += j * (vertices[2, 1] + params['delta'])  # Translate along the Y axis
                    translated[:, 2] += k * (vertices[5, 2] - vertices[4,2])    # Translate along the Z axis and match the points
                    
                    # if j==0 or j==6:
                    #     translated[:, 0] += 2.0
                    # if j==3:
                    #     translated[:, 0] += -2.0
                    
                    connMat_temp = -connMat
                    # Add unique vertices to the dictionary, keeping the smallest values
                    for vertex, node_num in zip(translated,range(translated.shape[0])):
                        vertice_loc = np.argwhere((np.sum(np.isclose(vertex,all_vertices),axis=1)==all_vertices.shape[1])==1)
                        vertice_size = vertice_loc.size
                        connMat_loc = np.argwhere(connMat_temp==-node_num)
                        if vertice_size == 0: 
                            all_vertices = np.append(all_vertices,vertex.reshape((1,-1)),axis=0)
                            connMat_temp[connMat_loc[:,0],connMat_loc[:,1]]=all_vertices.shape[0]-1
                        else:
                            # print(connMat[connMat_loc[0,:],connMat_loc[1,:]])
                            connMat_temp[connMat_loc[:,0],connMat_loc[:,1]]=vertice_loc
                            
                    all_connMat.append(connMat_temp)
        
        # # Convert the dictionary values to a NumPy array
        # all_vertices = np.array(list(unique_vertices_dict.values()))
        
        # # Convert the adjusted connectivity matrix to a NumPy array
        all_connMat = np.array(all_connMat).reshape((-1,2))
          
        return all_vertices, all_connMat
    def generatekelvin3DMeshHexagon(self,params): # arrowhead inside a hexagon
        if params is None:
            params = {'base': 2.,'height':5., 'theta': 0, 'alpha':0,'epsValue':0.,'phi':0.0,'delta':0, 'nx': 3, 'ny':3, 'nz':1,'r1':0.4,'r2':0.7}
    
        vertices, connMat = self.kelvin3Dunit(params) # generates a unit cell which we repeat
        # vertices, connMat = self.cuboctahedron3Dunit(params)
        nx = params['nx'] # Number of repetitions along the X axis
        ny = params['ny']  # Number of repetitions along the Y axis
        nz = params['nz']  # Number of repetitions along the Z axis
          
        unique_vertices_dict = {}
          
        # Lists to store the indices of unique vertices and connectivity matrices
        unique_vertex_indices = []
        adjusted_connMat_list = []
        
        all_vertices = vertices
        all_connMat = connMat
        #move vertices to origin
        Mov1 = vertices.copy()
        Mov1[:,[0,1]] += -params['base']/2;
                    
        for k in range(nz):
            for j in range(ny):
                # add extra arrowheads per side     
                if (j<1) or j>=ny-1:
                    etr = 0;
                elif j<3 or j>= ny-3:
                    etr = 1;# this should be 1 for panther project, 0 for other reasons
                elif j<5 or j>= ny-5:
                    etr = 2;
                else:
                    etr = 3;
                    
                for i in range(nx+2*etr):
                    # Translate the vertices of the unit cell
                    angRad = np.deg2rad(params['phi']*(i+j))
                    c = np.cos(angRad)
                    s = np.sin(angRad)
                    RotM = np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                    
                    #rotate about origin
                    Mov2 = (RotM@Mov1.T).T
                    
                    #move the vertices back to location
                    Mov2[:,[0,1]] += params['base']/2;
                    
                    translated = Mov2;
                    translated[:, 0] += i * (params['base'] + params['delta']) - min(j,etr)*params['base']  # Translate along the X axis
                    translated[:, 1] += j * (params['base'] + params['delta'])  # Translate along the Y axis
                    translated[:, 2] += k * (params['height'])    # Translate along the Z axis and match the points
                    
                    # if j==0 or j==6:
                    #     translated[:, 0] += 2.0
                    # if j==3:
                    #     translated[:, 0] += -2.0
                    
                    if i+j+k != 0:
                      connMat_temp = -connMat
                      # Add unique vertices to the dictionary, keeping the smallest values
                      for vertex, node_num in zip(translated,range(translated.shape[0])):
                          vertice_loc = np.argwhere((np.sum(np.isclose(vertex,all_vertices),axis=1)==all_vertices.shape[1])==1)
                          vertice_size = vertice_loc.size
                          connMat_loc = np.argwhere(connMat_temp==-node_num)
                          
                          if vertice_size == 0: 
                              all_vertices = np.append(all_vertices,vertex.reshape((1,-1)),axis=0)
                              connMat_temp[connMat_loc[:,0],connMat_loc[:,1]]=all_vertices.shape[0]-1
                          else:
                              # print(connMat[connMat_loc[0,:],connMat_loc[1,:]])
                              # print(vertice_loc,connMat_loc,connMat_temp)
                              connMat_temp[connMat_loc[:,0],connMat_loc[:,1]]=vertice_loc
                              
                      sumTemp = np.sum(connMat_temp,axis=1)
                      # all_connMatTemp = all_connMat.copy()
                      allSum = np.sum(all_connMat,axis=1)
                      # print(sumTemp,allSum)
                      checkPoints = np.where(1.0*(sumTemp==allSum.reshape(-1,1))==1.0)# array0 of allcon, array1 for conntemp
                      # print(checkPoints)
                      removeElem = []
                      # finds by comparing [a,b] and [b,a] with all conn and saves the element to be removed number
                      for cp in range(len(checkPoints[0])):
                          cond1 = connMat_temp[checkPoints[1][cp],:] == all_connMat[checkPoints[0][cp],:]
                          cond2 = connMat_temp[checkPoints[1][cp],:] == np.flip(all_connMat[checkPoints[0][cp],:])
                          if np.sum(cond1)==2 or np.sum(cond2)==2:
                              removeElem.append(cp)
                      connMat_temp = np.delete(connMat_temp,checkPoints[1][removeElem],axis=0)
                      
                      all_connMat = np.concatenate((all_connMat,connMat_temp))
        
        return all_vertices,all_connMat
        
    def generateArrow3DMeshTri(self,params):
        if params is None:
          params = {'base': 2, 'theta': 60, 'alpha': 30, 'nx': 2, 'ny': 2, 'nz': 2} 
      
        vertices, connMat = arrow3DunitTri(params) # generates a unit cell which we repeat # type: ignore
        nx = params['nx'] # Number of repetitions along the X axis
        ny = params['ny']  # Number of repetitions along the Y axis
        nz = params['nz']  # Number of repetitions along the Z axis
      
        unique_vertices_dict = {}
      
        # Lists to store the indices of unique vertices and connectivity matrices
        unique_vertex_indices = []
        adjusted_connMat_list = []
    
        all_vertices = vertices
        all_connMat = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # Translate the vertices of the unit cell
                    translated = vertices.copy()
                    translated[:, 0] += i * vertices[1, 0]  # Translate along the X axis
                    translated[:, 1] += j * vertices[2, 1]  # Translate along the Y axis
                    translated[:, 2] += k * (vertices[4, 2] - vertices[3,2])  # Translate along the Z axis and match the points
            
                    connMat_temp = -connMat
                    # Add unique vertices to the dictionary, keeping the smallest values
                    for vertex, node_num in zip(translated,range(translated.shape[0])):
                        vertice_loc = np.argwhere((np.sum(np.isclose(vertex,all_vertices),axis=1)==all_vertices.shape[1])==1)
                        vertice_size = vertice_loc.size
                        connMat_loc = np.argwhere(connMat_temp==-node_num)
                        if vertice_size == 0: 
                            all_vertices = np.append(all_vertices,vertex.reshape((1,-1)),axis=0)
                            connMat_temp[connMat_loc[:,0],connMat_loc[:,1]]=all_vertices.shape[0]-1
                        else:
                            # print(connMat[connMat_loc[0,:],connMat_loc[1,:]])
                            connMat_temp[connMat_loc[:,0],connMat_loc[:,1]]=vertice_loc
                            
                    all_connMat.append(connMat_temp)
        
        # # Convert the dictionary values to a NumPy array
        # all_vertices = np.array(list(unique_vertices_dict.values()))
        
        # # Convert the adjusted connectivity matrix to a NumPy array
        all_connMat = np.array(all_connMat).reshape((-1,2))
          
        return all_vertices, all_connMat
    
    
    def merge_vertices_and_conn(self,vertices_list, connMat_list):
        all_vertices = np.vstack(vertices_list)
            
        # Find the unique vertices and map the original indices to new indices
        unique_vertices, unique_indices = np.unique(all_vertices.round(decimals=10), axis=0, return_inverse=True)
        
        if len(unique_vertices) == len(all_vertices):
            updated_connMats = []
            offset = 0
            for connMat, vertices in zip(connMat_list, vertices_list):
                updated_connMat = connMat + offset
                updated_connMats.append(updated_connMat)
                offset += len(vertices)
            
            # Combine the updated connectivity matrices
            all_connMat = np.vstack(updated_connMats)
            
        else:    
            all_vertices = unique_vertices 
            # Update the connectivity matrices with new indices
            updated_connMats = []
            offset = 0
            # print("Con", connMat_list)
            for connMat, vertices in zip(connMat_list,vertices_list):
                updated_connMat = unique_indices[offset:offset + len(vertices)][connMat]
                updated_connMats.append(updated_connMat)
                offset += len(vertices)
                # print("This=",updated_connMat)
                # print("HERE")

            
            # Combine the updated connectivity matrices
            combined_connMat = np.vstack(updated_connMats)
            
            # Remove duplicate connections
            all_connMat = np.unique(combined_connMat, axis=0)
            
            # Remove self-connections (rows where the two values are the same)
            all_connMat = all_connMat[all_connMat[:, 0] != all_connMat[:, 1]]
            
            # print(all_connMat)
            # import sys;sys.exit()

        return all_vertices,all_connMat

    def intersect_lines(self,P1, P2, P3, P4):
        # Calculate direction vectors
        d4321 = np.dot(P4 - P3, P2 - P1)
        d2121 = np.dot(P2 - P1, P2 - P1)
        d4343 = np.dot(P4 - P3, P4 - P3)
        d1343 = np.dot(P1 - P3, P4 - P3)
        d1321 = np.dot(P1 - P3, P2 - P1)
        
        denom = d2121 * d4343 - d4321 * d4321
        
        if np.abs(denom) < 1e-7:
            # Lines are parallel or coincident
            is_intersecting = False
            Pa = np.array([])
            Pb = np.array([])
        else:
            mua = (d1343 * d4321 - d1321 * d4343) / denom
            mub = (d1343 + mua * d4321) / d4343
            
            # Calculate intersection points
            Pa = P1 + mua * (P2 - P1)
            Pb = P3 + mub * (P4 - P3)
            
            # Check if the intersection points are the same and within segments
            if np.linalg.norm(Pa - Pb) < 1e-7:
                if (0 <= mua <= 1) and (0 <= mub <= 1):
                    # Check that intersection is not at endpoints
                    if not np.any(np.all(np.isclose(Pa, np.array([P1, P2, P3, P4]), atol=1e-7), axis=1)):
                        is_intersecting = True
                    else:
                        is_intersecting = False
                else:
                    is_intersecting = False
            else:
                is_intersecting = False
        
        return Pa, Pb, is_intersecting
    
    # Function to find the intersection and divide connections
    def find_and_divide_intersections(self,vertices, connMat):
        new_vertices = vertices.tolist()
        new_connMat = connMat.tolist()
        intersections = {}
    
        for i in range(len(connMat)):
            for j in range(i + 1, len(connMat)):
                v1, v2 = connMat[i]
                v3, v4 = connMat[j]
    
                # Extract coordinates of the vertices
                p1, p2 = vertices[v1], vertices[v2]
                p3, p4 = vertices[v3], vertices[v4]
    
                # Find the intersection point
                # if p1==p3 or p1==p4 or p2==p3 or p2==p4:
                #   is_intersecting=False
                # else:
                Pa, Pb, is_intersecting  = self.intersect_lines(p1, p2, p3, p4)
                
                # if i==1 and j==9:
                #   print(p1,p2,p3,p4)
                #   print(Pa)
                #   import sys;
                #   sys.exit()
    
                if is_intersecting:
                    intersection_point = tuple(Pa)
                    
                    if intersection_point not in intersections:
                        intersections[intersection_point] = len(new_vertices)
                        new_vertices.append(intersection_point)
    
                    intersection_idx = intersections[intersection_point]
                    
                    new_connMat.append([v1, intersection_idx])
                    new_connMat.append([intersection_idx, v2])
                    new_connMat.append([v3, intersection_idx])
                    new_connMat.append([intersection_idx, v4])
                    new_connMat.remove([v1,v2])
                    new_connMat.remove([v3,v4])
    
        return np.array(new_vertices), np.array(new_connMat)
        
def nodeToElement(connMat):

    numVertices = np.max(connMat) + 1
    
    def find_rows_with_number(arr, target_number):
    
        mask = np.any(arr == target_number, axis=1)
    
        rows_with_number = np.where(mask)[0]
    
        return rows_with_number
    
    
    nodeToElems = []
    
    for i in range(numVertices):
    
        target_number = i

        rows_with_target_number = find_rows_with_number(connMat, target_number).tolist()

        # print("Rows containing the number", target_number, ":", rows_with_target_number)

        nodeToElems.append(rows_with_target_number)
        
    return nodeToElems