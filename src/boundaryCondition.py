import numpy as np

def boundaryConditionEx(exampleCase,nodeXY,connectivity):
    exampleNo = int(exampleCase[0])
    forces_d = None # dummy force
    try:
        if exampleCase[1:] == 'fx':
            f_val = [1.0,0.0,0.0,0.0,0.0,0.0]
        elif exampleCase[1:] == 'fy':
            f_val = [0.0,1.0,0.0,0.0,0.0,0.0]
        elif exampleCase[1:] == 'fz':
            f_val = [0.0,0.0,1.0,0.0,0.0,0.0]
        elif exampleCase[1:] == 'mx':
            f_val = [0.0,0.0,0.0,1.0,0.0,0.0]
        elif exampleCase[1:] == 'my':
            f_val = [0.0,0.0,0.0,0.0,1.0,0.0]
        elif exampleCase[1:] == 'mz':
            f_val = [0.0,0.0,0.0,0.0,0.0,1.0]
    except Exception as e:
        print("Taking loading z direction as default.", e)
        f_val = [0.0,0.0,1.0,0.0,0.0,0.0]
    
    if (exampleNo == 1): # single beam cantilever
       
        NodesFix = np.array([0.0]).astype(int)
        NodeForce = np.array([1.0]).astype(int)
            
        UX = np.ones((1,2))
        UX[:,1] = 0;
        UY = UX.copy()
        UT1 = UX.copy()
        UZ = UX.copy()
        UT2 = UX.copy()
        UT3 = UX.copy()
    
        fixtures = {'nodes':   NodesFix, \
                    'ux':      UX,\
                    'uy':      UY, \
                    'uz':      UZ,\
                    'utheta1':  UT1,\
                    'utheta2':  UT2,\
                    'utheta3':  UT3} # u_i = [fixed/free,disp_value] where fixed =1,free=0
        

        
        forces = {'nodes':NodeForce, 
        'fx':f_val[0]*np.array([2.0]), 
        'fy':f_val[1]*np.array([0.1]),
        'fz':f_val[2]*np.array([0.1]),
        'mx':f_val[3]*np.array([1.0]),
        'my':f_val[4]*np.array([0.75]),
        'mz':f_val[5]*np.array([0.75])}
        
        
        
    if (exampleNo == 2): # 3D beam bent
       
        NodesFix = np.array([0.0]).astype(int)
        NodeForce = np.array([3.0]).astype(int)
            
        UX = np.ones((1,2))
        UX[:,1] = 0;
        UY = UX.copy()
        UT1 = UX.copy()
        UZ = UX.copy()
        UT2 = UX.copy()
        UT3 = UX.copy()
    
        fixtures = {'nodes':   NodesFix, \
                    'ux':      UX,\
                    'uy':      UY, \
                    'uz':      UZ,\
                    'utheta1':  UT1,\
                    'utheta2':  UT2,\
                    'utheta3':  UT3} # u_i = [fixed/free,disp_value] where fixed =1,free=0

        forces = {'nodes':NodeForce, 
        'fx':f_val[0]*np.array([0.1]), 
        'fy':f_val[1]*np.array([0.1]),
        'fz':f_val[2]*np.array([0.1]),
        'mx':f_val[3]*np.array([2.0]),
        'my':f_val[4]*np.array([2.0]),
        'mz':f_val[5]*np.array([2.0])}
    
        
    if (exampleNo == 3): # Lattice C4, 6, 8, 12
        exampleName = 'C'
        
        base = np.where(nodeXY[:,2] == 0) # z==0
        nbase = np.shape(base)[1]
        
        top = np.where(nodeXY[:,2] == np.max(nodeXY[:,2])) #z==max
        ntop = np.shape(top)[1]
        # print(top)
        left = np.setdiff1d(np.where(nodeXY[:,0] == np.max(nodeXY[:,0])),base)
        # print(nodeXY[left,:],left) #3
        nleft = np.shape(left)[0]
        right = np.setdiff1d(np.where(nodeXY[:,0] == np.min(nodeXY[:,0])),base)
        nright = np.shape(right)[0]
        
        front = np.setdiff1d(np.where(nodeXY[:,1] == np.min(nodeXY[:,1])),base)
        nfront = np.shape(front)[0]
        back = np.setdiff1d(np.where(nodeXY[:,1] == np.max(nodeXY[:,1])),base)
        nback = np.shape(back)[0]
        # print(nodeXY[back,:],back) #0
        # apply load to center of 3x3 array
        # top = np.intersect1d(np.where(nodeXY[:,0] >= 12.7),top)
        # top = np.intersect1d(np.where(nodeXY[:,0] <= 2*12.7),top)
        # top = np.intersect1d(np.where(nodeXY[:,1] >= 12.7),top)
        # top = np.intersect1d(np.where(nodeXY[:,1] <= 2*12.7),top)
        # top = (top,)
        # ntop = np.shape(top)[1]
        
        NodesFix = base[0] #np.append(base) # skip top
        # print(NodesFix)
        FixSides = False;
        
        if FixSides:
            # print(NodesFix.shape) #80 size
            NodesFix = np.concatenate([NodesFix,left])
            NodesFix = np.concatenate([NodesFix,right])
            # print(NodesFix.shape) #120
            NodesFix = np.concatenate([NodesFix,back])
            NodesFix = np.concatenate([NodesFix,front])
            
        UX = np.ones((nbase,2))
        UX[:,1] = 0;
        UY = UX.copy()
        UT1 = UX.copy()
        UZ = UX.copy()
        UT2 = UX.copy()
        UT3 = UX.copy()
    
        def fixNodeValue(nNode,nValue,DirectionArray):
            U2 = np.zeros((nNode,2))
            if nValue.size != 0:
                U2[:,1] = nValue;
                U2[:,0] = 1; # fix x for middle top node / all nodes too
            DirectionArray = np.append(DirectionArray, U2,axis=0)
            return DirectionArray
        
        # UX = fixNodeValue(ntop,np.array([0.]),UX)
        # UY = fixNodeValue(ntop,np.array([0.]),UY)
        # uzVal = -0.1*np.max(nodeXY[:,2])
        # UZ = fixNodeValue(ntop,uzVal,UZ)
        # UT1 = fixNodeValue(ntop,np.array([]),UT1)
        # UT2 = fixNodeValue(ntop,np.array([]),UT2)
        # UT3 = fixNodeValue(ntop,np.array([]),UT3)

        if FixSides:
            UX = fixNodeValue(nleft,np.array([0.]),UX)
            UY = fixNodeValue(nleft,np.array([]),UY)
            UZ = fixNodeValue(nleft,np.array([]),UZ)
            UT1 = fixNodeValue(nleft,np.array([]),UT1)
            UT2 = fixNodeValue(nleft,np.array([]),UT2)
            UT3 = fixNodeValue(nleft,np.array([]),UT3)
            
            UX = fixNodeValue(nright,np.array([0.]),UX)
            UY = fixNodeValue(nright,np.array([]),UY)
            UZ = fixNodeValue(nright,np.array([]),UZ)
            UT1 = fixNodeValue(nright,np.array([]),UT1)
            UT2 = fixNodeValue(nright,np.array([]),UT2)
            UT3 = fixNodeValue(nright,np.array([]),UT3)
            
            UX = fixNodeValue(nback,np.array([]),UX)
            UY = fixNodeValue(nback,np.array([0.]),UY)
            UZ = fixNodeValue(nback,np.array([]),UZ)
            UT1 = fixNodeValue(nback,np.array([]),UT1)
            UT2 = fixNodeValue(nback,np.array([]),UT2)
            UT3 = fixNodeValue(nback,np.array([]),UT3)
            
            UX = fixNodeValue(nfront,np.array([]),UX)
            UY = fixNodeValue(nfront,np.array([0.]),UY)
            UZ = fixNodeValue(nfront,np.array([]),UZ)
            UT1 = fixNodeValue(nfront,np.array([]),UT1)
            UT2 = fixNodeValue(nfront,np.array([]),UT2)
            UT3 = fixNodeValue(nfront,np.array([]),UT3)
            
            #hardcode one node fix in y and x
            # if UY.shape[0]>80:
            #     uuyn = np.where(NodesFix==15)
            #     uuxn = np.where(NodesFix==18)
            #     # print(uuyn,uuxn)
            #     UY[uuyn[0],0] = 1
            #     UX[uuxn[0],0] = 1

        fixtures = {'nodes':   NodesFix, \
                    'ux':      UX,\
                    'uy':      UY, \
                    'uz':      UZ,\
                    'utheta1':  UT1,\
                    'utheta2':  UT2,\
                    'utheta3':  UT3} # u_i = [fixed/free,disp_value] where fixed =1,free=0

        # print(top,base)
        top =  top[0]
        forces = {'nodes':top, 
        'fx':f_val[0]*np.array([5.])/len(top)+top*0.0, 
        'fy':f_val[1]*np.array([5.])/len(top)+top*0.0, 
        'fz':f_val[2]*np.array([-20.])/len(top)+top*0.0, 
        'mx':f_val[3]*np.array([10.])/len(top)+top*0.0, 
        'my':f_val[4]*np.array([0.])/len(top)+top*0.0, 
        'mz':f_val[5]*np.array([0.])/len(top)+top*0.0}
    
    if (exampleNo == 4): # Lattice C4, 6, 8, 12
        exampleName = 'C'
        
        base = np.where(nodeXY[:,2] == 0) # z==0
        nbase = np.shape(base)[1]
        
        x_zero = np.where(nodeXY[:,0] == np.min(nodeXY[:,0]))[0] 
        x_L = np.where(nodeXY[:,0] == np.max(nodeXY[:,0]))[0] 
        
        
        NodesFix = x_zero #np.append(base) # skip top
        
        UX = np.ones((len(NodesFix),2))
        UX[:,1] = 0;
        UY = UX.copy()
        UT1 = UX.copy()
        UZ = UX.copy()
        UT2 = UX.copy()
        UT3 = UX.copy()
    
        def fixNodeValue(nNode,nValue,DirectionArray):
            U2 = np.zeros((nNode,2))
            if nValue.size != 0:
                U2[:,1] = nValue;
                U2[:,0] = 1; # fix x for middle top node / all nodes too
            DirectionArray = np.append(DirectionArray, U2,axis=0)
            return DirectionArray
        

        fixtures = {'nodes':   NodesFix, \
                    'ux':      UX,\
                    'uy':      UY, \
                    'uz':      UZ,\
                    'utheta1':  UT1,\
                    'utheta2':  UT2,\
                    'utheta3':  UT3} # u_i = [fixed/free,disp_value] where fixed =1,free=0

        yvals = nodeXY[x_L, 1]
        ymax = np.max(yvals)
        zvals = nodeXY[x_L, 2]
        zmax = np.max(zvals)
        
        # top_mid = np.where(
        #     (yvals >= 0.25 * ymax) & 
        #     (yvals <= 0.75 * ymax) & 
        #     (zvals >= 0.25 * zmax) & 
        #     (zvals <= 0.75 * zmax)
        # )[0]        
        # load_nodes = top[top_mid]
        
        load_nodes = x_L.copy()
        # print("load_nodes", load_nodes)
        # print("top nodes", top)
        forces = {'nodes':load_nodes, 
        'fx':f_val[0]*np.array([-2.])/len(load_nodes)+load_nodes*0.0, 
        'fy':f_val[1]*np.array([-2.])/len(load_nodes)+load_nodes*0.0, 
        'fz':f_val[2]*np.array([-2.])/len(load_nodes)+load_nodes*0.0, 
        'mx':f_val[3]*np.array([10.])/len(load_nodes)+load_nodes*0.0, 
        'my':f_val[4]*np.array([0.])/len(load_nodes)+load_nodes*0.0, 
        'mz':f_val[5]*np.array([0.])/len(load_nodes)+load_nodes*0.0}    
        
    if (exampleNo == 5): # Lattice C4, 6, 8, 12
        exampleName = 'MBB'
        
        base = np.where(nodeXY[:,2] == 0) # z==0
        nbase = np.shape(base)[1]
        
        x_zero = np.where(nodeXY[:,0] == np.min(nodeXY[:,0]))[0] 
        x_L = np.where(nodeXY[:,0] == np.max(nodeXY[:,0]))[0] 
        
        z_zero = np.where(nodeXY[:,2] == np.min(nodeXY[:,2]))[0] 
        z_H = np.where(nodeXY[:,2] == np.max(nodeXY[:,2]))[0]
        
        # print("z_zero", z_zero)
        # print("z_H", z_H)
        
        xvals = nodeXY[z_zero, 0]
        xmax = np.max(xvals)
        xmin = np.min(xvals)
        xvals_scaled = (xvals - xmin) / (xmax - xmin)
        Bot_sides = np.where(
            (xvals_scaled <= 0.11) | 
            (xvals_scaled >= 0.89)
        )[0]     
        
        NodesFix = z_zero[Bot_sides] #np.append(base) # skip top
                    
        UX = np.ones((len(NodesFix),2))
        UX[:,1] = 0;
        UY = UX.copy()
        UT1 = UX.copy()
        UZ = UX.copy()
        UT2 = UX.copy()
        UT3 = UX.copy()
    
        def fixNodeValue(nNode,nValue,DirectionArray):
            U2 = np.zeros((nNode,2))
            if nValue.size != 0:
                U2[:,1] = nValue;
                U2[:,0] = 1; # fix x for middle top node / all nodes too
            DirectionArray = np.append(DirectionArray, U2,axis=0)
            return DirectionArray
        
        
            
        fixtures = {'nodes':   NodesFix, \
                    'ux':      UX,\
                    'uy':      UY, \
                    'uz':      UZ,\
                    'utheta1':  UT1,\
                    'utheta2':  UT2,\
                    'utheta3':  UT3} # u_i = [fixed/free,disp_value] where fixed =1,free=0

        
        xvals = nodeXY[z_H, 0]
        xmax = np.max(xvals)
        xmin = np.min(xvals)
        xvals_scaled = (xvals - xmin) / (xmax - xmin)
        top_mid = np.where(
            (xvals_scaled >= 0.4) & 
            (xvals_scaled <= 0.6)
        )[0]   
        
        load_nodes = z_H[top_mid]
        # print("load_nodes", load_nodes)
        # print("top nodes", top)
        forces = {'nodes':load_nodes, 
        'fx':f_val[0]*np.array([-10.])/len(load_nodes)+load_nodes*0.0, 
        'fy':f_val[1]*np.array([-10.])/len(load_nodes)+load_nodes*0.0, 
        'fz':f_val[2]*np.array([-10.])/len(load_nodes)+load_nodes*0.0, 
        'mx':f_val[3]*np.array([10.])/len(load_nodes)+load_nodes*0.0, 
        'my':f_val[4]*np.array([0.])/len(load_nodes)+load_nodes*0.0, 
        'mz':f_val[5]*np.array([0.])/len(load_nodes)+load_nodes*0.0}
    
    if (exampleNo == 6): 
        exampleName = 'MCA' # micro compliant amplifier
        
        base = np.where(nodeXY[:,2] == 0) # z==0
        nbase = np.shape(base)[1]
        
        xvals = nodeXY[:, 0]
        xmax = np.max(xvals)
        xmin = np.min(xvals)
        xvals_scaled = (xvals - xmin) / (xmax - xmin)
        
        x_zero = np.where(xvals_scaled <= 0.17)[0] 
        x_L = np.where(xvals_scaled >= 1.0-0.17)[0] 
        
        zvals = nodeXY[:, 2]
        zmax = np.max(zvals)
        zmin = np.min(zvals)
        zvals_scaled = (zvals - zmin) / (zmax - zmin)
        
        z_zero = np.where(zvals_scaled == 0.0)[0] 
        z_H = np.where(zvals_scaled == 1.0)[0]
        
        Fix_bot = np.intersect1d(z_zero, np.concatenate((x_zero,x_L)))
        Fix_top = np.intersect1d(z_H, np.concatenate((x_zero,x_L)))
        
        NodesFix = Fix_bot #np.concatenate((Fix_bot, Fix_top))
                    
        UX = np.ones((len(NodesFix),2))
        UX[:,1] = 0;
        UY = UX.copy()
        UT1 = UX.copy()
        UZ = UX.copy()
        UT2 = UX.copy()
        UT3 = UX.copy()
    
        def fixNodeValue(nNode,nValue,DirectionArray):
            U2 = np.zeros((nNode,2))
            if nValue.size != 0:
                U2[:,1] = nValue;
                U2[:,0] = 1; # fix x for middle top node / all nodes too
            DirectionArray = np.append(DirectionArray, U2,axis=0)
            return DirectionArray
        
        # lets fix all the nodes z values 
        allNodesFixY = np.setdiff1d(np.arange(nodeXY.shape[0]), NodesFix)
        NodesFix =  np.concatenate((NodesFix, allNodesFixY))
        UX = fixNodeValue(len(allNodesFixY),np.array([]),UX)
        UY = fixNodeValue(len(allNodesFixY),np.array([0.0]),UY)
        UZ = fixNodeValue(len(allNodesFixY),np.array([]),UZ)
        UT1 = fixNodeValue(len(allNodesFixY),np.array([]),UT1)
        UT2 = fixNodeValue(len(allNodesFixY),np.array([]),UT2)
        UT3 = fixNodeValue(len(allNodesFixY),np.array([]),UT3)

        fixtures = {'nodes':   NodesFix, \
                    'ux':      UX,\
                    'uy':      UY, \
                    'uz':      UZ,\
                    'utheta1':  UT1,\
                    'utheta2':  UT2,\
                    'utheta3':  UT3} # u_i = [fixed/free,disp_value] where fixed =1,free=0

        
        xvals = nodeXY[z_zero, 0]
        xmax = np.max(xvals)
        xmin = np.min(xvals)
        xvals_scaled = (xvals - xmin) / (xmax - xmin)
        top_mid = np.where(
            (xvals_scaled >= 0.4) & 
            (xvals_scaled <= 0.6)
        )[0]   
        
        load_nodes = z_zero[top_mid]
        
        ff = 1.0
        forces = {'nodes':load_nodes, 
        'fx':f_val[0]*np.array([ff])/len(load_nodes)+load_nodes*0.0, 
        'fy':f_val[1]*np.array([ff])/len(load_nodes)+load_nodes*0.0, 
        'fz':f_val[2]*np.array([ff])/len(load_nodes)+load_nodes*0.0, 
        'mx':f_val[3]*np.array([ff])/len(load_nodes)+load_nodes*0.0, 
        'my':f_val[4]*np.array([0.])/len(load_nodes)+load_nodes*0.0, 
        'mz':f_val[5]*np.array([0.])/len(load_nodes)+load_nodes*0.0}
        
        xvals = nodeXY[z_H, 0]
        xmax = np.max(xvals)
        xmin = np.min(xvals)
        xvals_scaled = (xvals - xmin) / (xmax - xmin)
        top_mid = np.where(
            (xvals_scaled >= 0.49) & 
            (xvals_scaled <= 0.51)
        )[0]   
                
        load_nodes_d = z_H[top_mid]
        fd = -1.0*len(load_nodes_d)
        forces_d = {'nodes':load_nodes_d, 
        'fx':f_val[0]*np.array([fd])/len(load_nodes_d)+load_nodes_d*0.0, 
        'fy':f_val[1]*np.array([fd])/len(load_nodes_d)+load_nodes_d*0.0, 
        'fz':f_val[2]*np.array([fd])/len(load_nodes_d)+load_nodes_d*0.0, 
        'mx':f_val[3]*np.array([fd])/len(load_nodes_d)+load_nodes_d*0.0, 
        'my':f_val[4]*np.array([0.])/len(load_nodes_d)+load_nodes_d*0.0, 
        'mz':f_val[5]*np.array([0.])/len(load_nodes_d)+load_nodes_d*0.0}
        
        
        
    ###############################################################################################    
    bc = {'forces':forces, 'fixtures':fixtures,'forces_d':forces_d}
    
    return bc