import numpy as np
from dolfin import *
import fenics as fe
import matplotlib.pyplot as plt
import networkx as nx


def plot_mesh(coord,labels,edges,title):
    ''' Function to plot a mesh saved as a graph.
        INPUT: 
            coord : coordinates of the vertices7nodes of the network.
            labels : labels of the vertices of the network.
            edges : edges of the network.
            title : title of the plot.
    '''
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Define a color map
    color_map = 'viridis'

    # Create the scatter plot

    scatter_plot = ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=labels, cmap=color_map, s=50, edgecolor='k', alpha=0.8)

    # Plot edges
    for line in edges:
        line_array = np.array([coord[line[0]],coord[line[1]]])
        ax.plot(line_array[:, 0], line_array[:, 1], line_array[:, 2], color='r', linewidth=2, label='edges')

    # Customize the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    ax.set_xlim(-1, 1) 

    ax.set_ylim(-1, 1) 

    ax.set_zlim(-1, 1) 


    # Add color bar
    cbar = fig.colorbar(scatter_plot, ax=ax, pad=0.1)
    cbar.set_label('Markers')
    plt.show()

def compute_pressure(meshQ,inlet_points,outlet_points,p_in,p_out):
    ''' Function that computes the pressure in each vertex of the net, using a classical diffusion problem in a specific direction, solved with fenics.
        INPUT:
            meshQ : fenics mesh function of the net.
            inlet_points : coordinates of the inlet points.
            outlet_points : coordinates of the outlet points.
            p_in : Dirichlet condition for the inlet points.
            p_out : Dirichlet condition fot the outlet points.

        OUTPUT:
            sol : pressure solution in each vertex.
    '''
    
    W = FunctionSpace(meshQ,'P', 1)
    #BC
    def inlet(x):
        flag=False
        for p in inlet_points:
            if(x[0]==p[0] and x[1]==p[1] and x[2]==p[2]):
                flag=True
                break

        return flag

    def outlet(x):
        flag=False
        for p in outlet_points:
            if(x[0]==p[0] and x[1]==p[1] and x[2]==p[2]):
                flag=True
                break

        return flag

    bc_in = DirichletBC(W, p_in,inlet)

    bc_out = DirichletBC(W, p_out, outlet)

    p = TrialFunction(W)
    q = TestFunction(W)
    f = Constant(0)
    a = inner(grad(p), grad(q))*dx
    L = f*q*dx

    sol=Function(W)
    bcs=[bc_in,bc_out]
    solve(a == L, sol, bcs)
    return sol

def mu_bl(D):
    #Blood viscosity

    mu_p=1*10**(-3) # plasma viscosity Pa*s
    H=0.45 #discharge hematocrit
    C=(0.8 + np.exp(-0.075*D))*(-1+1/(1+10**(-11)*D**(12)))+ 1/(1+10**(-11)*D**(12))# influemce of H on mu_bl
    mu_45=6*np.exp(-0.085*D)+3.2 - 2.44 *np.exp(-0.06 * D**0.645)
    res=mu_p * (1 + (mu_45-1)*((1-H)**C-1)/((1-0.45)**C-1) * (D/(D-1.1))**2) * (D/(D-1.1))**2
    return res

def compute_VF_mu_up(meshQ,outlet_indexes,sol_val,coord,edges,radii):
    #Function that computes volume flux at the outlet and the upscaled viscosity

    VF_out=0
    mu_bl_up=0
    iter=0
    for indx in outlet_indexes:
        iter=iter+1
        res=np.where(edges[:,1]==indx)[0]
        prev=edges[res][0][0]
        Rk=radii[res][0]
        p_out=sol_val[indx]
        p_prev=sol_val[prev]
        dis=np.linalg.norm(coord[indx]*1e-3-coord[prev]*1e-3)
        dgrad=(p_out-p_prev)/np.abs((dis))

        VF_out = VF_out +  (np.pi*Rk**4)/(8*mu_bl(2*Rk/(1e-6))) *dgrad
    for Rk in radii:  
         mu_bl_up= mu_bl_up + mu_bl(2*Rk/(1e-6))
    mu_bl_up=mu_bl_up/len(edges)
    return VF_out,mu_bl_up,iter

def G(i,j,meshQ,coord,Rij):
    #Conductance associated with the vessel/edge (i,j)

    Lij=np.linalg.norm(coord[i]*1e-3-coord[j]*1e-3)
    res = (np.pi*Rij**4)/(8*mu_bl(2*Rij)*Lij)
    return  res

def compute_pressure2(dir,meshQ,edges,coord,labels,radii,inlet_indexes,inlet_indexes_dir,outlet_indexes,outlet_indexes_dir,F_in,F_out,p_in,p_out):
    ''' Function that computes the pressure in each vertex , by solving the linear system obtained studying the equilibrium condition at each bifurcation point.
        INPUT:
            dir : direction of the flow (1: x , 2: y , 3: z).
            meshQ : fenics mesh function of the net.
            edges : edges of the network.
            coord : coordinates of the vertices7nodes of the network.
            labels : labels of the vertices of the network.
            radii : radii of the edges.
            inlet_indexes : indexes of all the inlet points.
            inlet_indexes_dir : indexes of the inlet points in the chosen direction.
            outlet_indexes : indexes of all the outlet points.
            outlet_indexes_dir : indexes of the outlet points in the chosen direction.
            F_in : labels associated with each inlet points identifyng the face of belonging. (Ex: the label '1' is associated the the inlet point on the inlet face in the x direction.)
            F_out : labels associated with each outlet points identifyng the face of belonging.
            p_in : Dirichlet condition for the inlet points.
            p_out : Dirichlet condition fot the outlet points.

        OUTPUT:
            ptot : pressure solution in each vertex.
    
    '''

    n_size=np.shape(coord)[0]

    M= np.zeros((n_size, n_size),dtype=float)
    q=np.zeros((n_size,1),dtype=float)

    #per come sono costruiti gli edges, edge[0] è sempre con label 555
    count_in=0
    count_out=0
    for rr,edge in enumerate(edges):
        i=edge[0] 
        j=edge[1]
        Rij=radii[rr]
        val=float(G(i,j,meshQ,coord,Rij))
        if labels[j]==111:
            ind=np.where(inlet_indexes==j)[0]
            if F_in[ind]==dir:
                count_in=count_in+1
                q[i]=q[i]+p_in*val
                M[i,i]=M[i,i]+val
            else:
                M[i,j]=-val
                M[j,i]=-1
                M[i,i]=M[i,i]+val
                M[j,j]=M[j,j]+1
            continue   

        if labels[j]==999:
            ind=np.where(outlet_indexes==j)[0]
            if F_out[ind]==dir:
                count_out=count_out+1
                q[i]=q[i]+p_out*val
                M[i,i]=M[i,i]+val
            else:
                M[i,j]=-val
                M[j,i]=-1
                M[i,i]=M[i,i]+val
                M[j,j]=M[j,j]+1
            continue
        
        M[i,j]=-val
        M[j,i]=-val
        M[i,i]=M[i,i]+val
        M[j,j]=M[j,j]+val
    
    #creating reduced matrices
    non_zero_rows = np.any(M != 0, axis=1)
    #print(np.shape(np.where(non_zero_rows==False)[0]))

    non_zero_cols = np.any(M != 0, axis=0)
    #print(np.shape(np.where(non_zero_cols==False)[0]))


    M_inter = M[non_zero_rows, :]
    M_reduced=M_inter[:,non_zero_cols]
    q_reduced = q[non_zero_rows]

    #print(np.shape(M_reduced),np.shape(q_reduced))
    p = np.linalg.solve(M_reduced, q_reduced)
    # print(np.linalg.cond(M_reduced))
    # print(np.allclose(M_reduced, M_reduced.T))
    
    PIN=p_in*np.ones((np.shape(inlet_indexes_dir)[0],))
    POUT=p_out*np.ones((np.shape(outlet_indexes_dir)[0],))

   
    p_tot=np.zeros((n_size,),dtype=float)
    p_tot[non_zero_rows]=p.flatten()
    p_tot[inlet_indexes_dir]=PIN
    p_tot[outlet_indexes_dir]=POUT

    return p_tot


def compute_k(dir,meshQ,radii,outlet_indexes_dir,p_dir,coord,edges,p_in,p_out):
    #Computes the component of the permeability tensor of the chosen direction

    VF_out,mu_bl_up,iter=compute_VF_mu_up(meshQ,outlet_indexes_dir,p_dir,coord,edges,radii)

    Lx=np.abs(np.min(coord[:,0])-np.max(coord[:,0]))*1e-3
    Ly=np.abs(np.min(coord[:,1])-np.max(coord[:,1]))*1e-3
    Lz=np.abs(np.min(coord[:,2])-np.max(coord[:,2]))*1e-3

    if dir==1:
        k=(VF_out*mu_bl_up*Lx)/(Ly*Lz*(p_in-p_out))
    if dir==2:
        k=(VF_out*mu_bl_up*Ly)/(Lx*Lz*(p_in-p_out))
    if dir==3:
        k=(VF_out*mu_bl_up*Lz)/(Ly*Lx*(p_in-p_out))

    return k, mu_bl_up

def compute_C_value(meshQ,edges,coord,radii,xmax,xmin,ymax,ymin,zmax,zmin):
    #Computes the value of the constant associated with the RHS term

    rho_int=1000
    L_cap=1e-12
    S=0
    for rr,edge in enumerate(edges):
        Rk=radii[rr]
        S=S+np.linalg.norm(coord[edge[0]]*1e-3-coord[edge[1]]*1e-3)*2*Rk*np.pi
    vol_REV = np.abs(xmax-xmin)*1e-3*np.abs(ymax-ymin)*1e-3*np.abs(zmax-zmin)*1e-3
    C_j = (rho_int * L_cap * S)/vol_REV
    return C_j


def find_intersection2(p1,p2,x_planes,y_planes,z_planes):
    ''' Function that computes the itersections between the edge connecting p1 to p2 and all the possible planes originated from the REV division in the x,y,z direction, saving only the feasible ones. 
        INPUT:
            p1 : starting points of the edge.
            p2 : final point of the edge.
            x_planes : division planes in the x direction.
            y_planes : division planes in the y direction.
            z_planes : division planes in the z direction.

        OUTPUT:
            res_final : coordinates of the feasible intersection points.

    '''
    def line3D(p1,p2):
        a=p2[0]-p1[0]
        b=p2[1]-p1[1]
        c=p2[2]-p1[2]
        return a,b,c
    
    def compute_inter(A,plane,p1,p2):
        b=np.copy(p1)
        b = np.append(b, plane)
        res=np.linalg.solve(A,b)
        res_final=[]
        if (res[0] < np.maximum(p1[0], p2[0]) and
            res[0] > np.minimum(p1[0], p2[0]) and
            res[1] < np.maximum(p1[1], p2[1]) and
            res[1] > np.minimum(p1[1], p2[1]) and
            res[2] < np.maximum(p1[2], p2[2]) and
            res[2] > np.minimum(p1[2], p2[2])):
            res_final=res[:3]
        return res_final
    a,b,c=line3D(p1,p2)
    #print(a,b,c)
    tol=1e-5

    res_final=[]
    Ax=[[1,0,0,-a],[0,1,0,-b],[0,0,1,-c],[1,0,0,0]]
    Ay=[[1,0,0,-a],[0,1,0,-b],[0,0,1,-c],[0,1,0,0]]
    Az=[[1,0,0,-a],[0,1,0,-b],[0,0,1,-c],[0,0,1,0]]

    #inter with x_planes
    if np.abs(a)>tol:
        for xp in x_planes:
            res=compute_inter(Ax,xp,p1,p2)
            if(len(res)!=0):
                res_final.append(res)

    #inter with y_planes
    if np.abs(b)>tol:
        for yp in y_planes:
            res=compute_inter(Ay,yp,p1,p2)
            if(len(res)!=0):
                res_final.append(res)
        
    #inter with z_planes
    if np.abs(c)>tol:
        for zp in z_planes:
            res=compute_inter(Az,zp,p1,p2)
            if(len(res)!=0):
                res_final.append(res)

    res_final=np.array(res_final)
    return res_final

def create_mesh_REV(name,coord_rev,edges_rev):
    '''Function that creates a graph mesh given coordinates and edges.
        INPUT:
            name : mesh name.
            coord_rev : coordinates of the new mesh.
            edges_rev : edges of the new mesh.
    '''

    # Define mesh vertices for a 3D capillary bed
    vertices = coord_rev

    # Define mesh cells (line elements)
    cells = edges_rev
    # Create a mesh
    mesh = Mesh()
    with XDMFFile(name+".xdmf") as xdmf:
        editor = MeshEditor()
        editor.open(mesh, 'interval', 1, 3)  # 'interval' for a 3D mesh
        editor.init_vertices(len(vertices))

        # Add vertices
        for i, vertex in enumerate(vertices):
            editor.add_vertex(i, vertex)

        # Add cells
        editor.init_cells(len(cells))
        for i, cell in enumerate(cells):
            editor.add_cell(i,cell)

        editor.close()

        xdmf.write(mesh)

    print(f"Number of vertices: {mesh.num_vertices()}")
    print(f"Number of cells: {mesh.num_cells()}")


def compute_faces(labels,coord,xmin,xmax,ymin,ymax,zmin,zmax):
    ''' Function the associates the face labels to the respective inlet and outlet points.
        INPUT: 
            labels : labels of the vertices of the network.
            coord :  coordinates of the vertices/nodes of the network.
            xmin : left limit of the mesh in the x direction.
            xmax : right limit of the mesh in the x direction.
            ymin : left limit of the mesh in the y direction.
            ymax : right limit of the mesh in the y direction.
            zmin : left limit of the mesh in the z direction.
            zmax : right limit of the mesh in the z direction.

        OUTPUT:
            points : dictionary that contains the inlet/outlet points grouped by their belonging to each one of the faces.
            indexes : dictionary that contains the inlet/outlet indexes grouped by their belonging to each one of the faces.
            F_in : labels associated with each inlet points identifyng the face of belonging. (Ex: the label '1' is associated the the inlet point on the inlet face in the x direction.)
            F_out : labels associated with each outlet points identifyng the face of belonging.
    '''
    points={}
    indexes={}

    outlet_indexes=np.where(labels==999)[0]
    inlet_indexes=np.where(labels==111)[0]
    outlet_points=coord[outlet_indexes]
    inlet_points=coord[inlet_indexes]

    tol=1e-2/8

    F_out=np.zeros((np.shape(outlet_points)[0],1))

    ind1_out=np.where(np.abs(outlet_points[:,0]-xmax)<tol)[0]
    F_out[ind1_out]=1

    ind2_out=np.where(np.abs(outlet_points[:,1]-ymax)<tol)[0]
    F_out[ind2_out]=2
   
    ind3_out=np.where(np.abs(outlet_points[:,2]-zmax)<tol)[0]
    F_out[ind3_out]=3

    outlet_points_x=outlet_points[np.where(F_out==1)[0]]
    outlet_points_y=outlet_points[np.where(F_out==2)[0]]
    outlet_points_z=outlet_points[np.where(F_out==3)[0]]

    outlet_indexes_x=outlet_indexes[np.where(F_out==1)[0]]
    outlet_indexes_y=outlet_indexes[np.where(F_out==2)[0]]
    outlet_indexes_z=outlet_indexes[np.where(F_out==3)[0]]

    F_in=np.zeros((np.shape(inlet_points)[0],1))

    ind1_in=np.where(np.abs(inlet_points[:,0]-xmin)<tol)[0]
    F_in[ind1_in]=1

    ind2_in=np.where(np.abs(inlet_points[:,1]-ymin)<tol)[0]
    F_in[ind2_in]=2

    ind3_in=np.where(np.abs(inlet_points[:,2]-zmin)<tol)[0]
    F_in[ind3_in]=3

    inlet_points_x=inlet_points[np.where(F_in==1)[0]]
    inlet_points_y=inlet_points[np.where(F_in==2)[0]]
    inlet_points_z=inlet_points[np.where(F_in==3)[0]]

    inlet_indexes_x=inlet_indexes[np.where(F_in==1)[0]]
    inlet_indexes_y=inlet_indexes[np.where(F_in==2)[0]]
    inlet_indexes_z=inlet_indexes[np.where(F_in==3)[0]]

    points['inlet']={'x': inlet_points_x}
    points['inlet']['y']=inlet_points_y
    points['inlet']['z']=inlet_points_z
    points['inlet']['total']=inlet_points


    points['outlet']={'x': outlet_points_x}
    points['outlet']['y']=outlet_points_y
    points['outlet']['z']=outlet_points_z
    points['outlet']['total']=outlet_points


    indexes['inlet']={'x': inlet_indexes_x}
    indexes['inlet']['y']=inlet_indexes_y
    indexes['inlet']['z']=inlet_indexes_z
    indexes['inlet']['total']=inlet_indexes


    indexes['outlet']={'x': outlet_indexes_x}
    indexes['outlet']['y']=outlet_indexes_y
    indexes['outlet']['z']=outlet_indexes_z
    indexes['outlet']['total']=outlet_indexes

    return points,indexes,F_in,F_out

def compute_connected_components(edges):
    #Computes the connecte components of a graph
    G = nx.Graph()
    G.add_edges_from(edges)
    connected_components = list(nx.connected_components(G))

    for i, component in enumerate(connected_components, 1):
        component=np.array(list(component))
        edges=G.subgraph(component).edges()

    return G,connected_components

def compute_REV_labels(mesh,n_div_x,n_div_y,n_div_z):
    ''' Function that assign each coordinates of the graph to a REV, considering a cubic region. The results are equal subcubes.
        INPUT: 
            mesh : graph saved as a fenics mesh type.
            n_div_x : number of division in the x direction.
            n_div_y : number of division in the y direction.
            n_div_z : number of division in the z direction.
        
        OUTPUT: 
            REV_labels :  dictionary that contains the points grouped by their REV.
    
    '''
    print(f'Total REV: {n_div_x*n_div_y*n_div_z}')

    coord=mesh.coordinates()

    REV_labels=np.zeros(np.shape(coord)[0],)
    count=1
    tol2=1e-15
    xmin=np.min(coord[:,0])
    xmax=np.max(coord[:,0])
    ymin=np.min(coord[:,1])
    ymax=np.max(coord[:,1])
    zmin=np.min(coord[:,2])
    zmax=np.max(coord[:,2])

    x= np.linspace(xmin,xmax,n_div_x+1)
    y= np.linspace(ymin,ymax,n_div_y+1)
    z=np.linspace(zmin,zmax,n_div_z+1)

    for i in np.arange(len(x)-1):
        for j in np.arange(len(y)-1):
            for k in np.arange(len(z)-1):
                xmin_t=np.maximum(xmin,x[i])
                xmax_t=np.minimum(xmax,x[i+1])
                ymin_t=np.maximum(ymin,y[j])
                ymax_t=np.minimum(ymax,y[j+1])
                zmin_t=np.maximum(zmin,z[k])
                zmax_t=np.minimum(zmax,z[k+1])

                indexes=np.where(((coord[:,0]-xmin_t)>=-tol2) & ((coord[:,0]-xmax_t)<=tol2) &
                    ((coord[:,1]-ymin_t)>=-tol2) & ((coord[:,1]-ymax_t)<=tol2) &
                    ((coord[:,2]-zmin_t)>=-tol2) & ((coord[:,2]-zmax_t)<=tol2))[0]
            

                REV_labels[indexes]=count
                count=count+1
    return REV_labels

def compute_REV_labels_TOTAL(mesh,n_div_x,n_div_y,n_div_z,x,y,z):
    ''' Function that assign each coordinates of the graph to a REV, considering a cubic region. The resulting sections can be not equal subcubes, depending on the region delimiters given as input. This is ueful for mesh that are more dense in specific region.
        INPUT: 
            mesh : graph saved as a fenics mesh type.
            n_div_x : number of division in the x direction.
            n_div_y : number of division in the y direction.
            n_div_z : number of division in the z direction.
            x : delimiters of the sections in the x direction.
            y : delimiters of the sections in the y direction.
            z : delimiters of the sections in the z direction.
        
        OUTPUT: 
            REV_labels :  dictionary that contains the points grouped by their REV.
    
    '''
    print(f'Total REV: {n_div_x*n_div_y*n_div_z}')

    coord=mesh.coordinates()

    REV_labels=np.zeros(np.shape(coord)[0],)
    count=1
    tol2=1e-15
    xmin=np.min(coord[:,0])
    xmax=np.max(coord[:,0])
    ymin=np.min(coord[:,1])
    ymax=np.max(coord[:,1])
    zmin=np.min(coord[:,2])
    zmax=np.max(coord[:,2])

    for i in np.arange(len(x)-1):
        for j in np.arange(len(y)-1):
            for k in np.arange(len(z)-1):
                xmin_t=np.maximum(xmin,x[i])
                xmax_t=np.minimum(xmax,x[i+1])
                ymin_t=np.maximum(ymin,y[j])
                ymax_t=np.minimum(ymax,y[j+1])
                zmin_t=np.maximum(zmin,z[k])
                zmax_t=np.minimum(zmax,z[k+1])

                indexes=np.where(((coord[:,0]-xmin_t)>=-tol2) & ((coord[:,0]-xmax_t)<=tol2) &
                    ((coord[:,1]-ymin_t)>=-tol2) & ((coord[:,1]-ymax_t)<=tol2) &
                    ((coord[:,2]-zmin_t)>=-tol2) & ((coord[:,2]-zmax_t)<=tol2))[0]
            

                REV_labels[indexes]=count
                count=count+1
    return REV_labels

def REV_division(name_mesh,dir_results,meshQ,radii, Q_markers,n_div_x,n_div_y,n_div_z,p_in,p_out,plot_mesh_flag=False, M_plot_flag=False,lap_plot_flag=False):
    '''Function that divides a mesh growing in a cubic region in a specified number of equal REV, creating the corresponding submeshes and computing the pressure in each of them, testing both the laplacian ad the linear system methods.
        INPUT:
            name_mesh : name of the mesh.
            dir_results : directory in with the resulting mesh are solved.
            meshQ : mesh of the network saved as fenics mesh type.
            radii : radii associated with the edges of the network.
            Q_markers : markers type that store mesh info.
            n_div_x : number of division in the x direction.
            n_div_y : number of division in the y direction.
            n_div_z : number of division in the z direction.
            p_in : Dirichlet condition for the inlet points.
            p_out : Dirichlet condition fot the outlet points.
            plot_mesh_flag : flag that enables the plot of the mesh (if TRUE it slows the code significantly)
            M_plot_flag : flag that enables the plot of the pressure results for each REV using the linear system method (if TRUE it slows the code significantly).
            lap_plot_flag : flag that enables the plot of the pressure results for each REV using the laplacian  method (if TRUE it slows the code significantly).

        OUTPUT: 
            K_x : values of the x component of the upscaled permeability tensor (one for each REV).
            K_y : values of the y component of the upscaled permeability tensor (one for each REV).
            K_z : values of the z component of the upscaled permeability tensor (one for each REV).
            C : values of the RHS upscaled constant (one for each REV).
            mu_bl_up: values of the upscaled blood viscosity (one for each REV).
    '''
    #NOTE THE DIVISION IN MADE FOR A CUBE OF DIM (-1,1)^3
    #Mesh Info
    coord=meshQ.coordinates()
    labels=Q_markers.array()
    edges=meshQ.cells()

    print('-------------------- MESH INFO --------------------')
    print(f'Coordinates: {np.shape(coord)}')
    print(f'Edges: {np.shape(edges)}')
    print(f'Radii: {np.shape(radii)}')
    print(f'Labels: {np.shape(labels)}')
    print(f'Interior Points: {np.shape(np.where(labels==555)[0])}')
    print(f'Inlet Points: {np.shape(np.where(labels==111)[0])}')
    print(f'Outlet Points: {np.shape(np.where(labels==999)[0])}')

    print()
    
    if plot_mesh_flag == True:
        plot_mesh(coord,labels,edges,'Capillary Net')

    print(f'Total REV number: {n_div_x*n_div_y*n_div_z}')

    #Division
    REV_division={}
    REV_labels=np.zeros(np.shape(labels))
    #np.shape(REV_labels)
    count=1
    sum=0
    tol2=1e-15
    xmin=-1
    xmax=1
    ymin=-1
    ymax=1
    zmin=-1
    zmax=1

    x= np.linspace(xmin,xmax,n_div_x+1)
    x_planes=x[1:-1]

    y= np.linspace(ymin,ymax,n_div_y+1)
    y_planes=y[1:-1]

    z=np.linspace(zmin,zmax,n_div_z+1)
    z_planes=z[1:-1]

    for i in np.arange(len(x)-1):
        for j in np.arange(len(y)-1):
            for k in np.arange(len(z)-1):
                REV_division[count]={'xmin':np.maximum(xmin,x[i])}
                REV_division[count]['xmax']=np.minimum(xmax,x[i+1])
                REV_division[count]['ymin']=np.maximum(ymin,y[j])
                REV_division[count]['ymax']=np.minimum(ymax,y[j+1])
                REV_division[count]['zmin']=np.maximum(zmin,z[k])
                REV_division[count]['zmax']=np.minimum(zmax,z[k+1])

                indexes=np.where(((coord[:,0]-REV_division[count]['xmin'])>=-tol2) & ((coord[:,0]-REV_division[count]['xmax'])<=tol2) &
                    ((coord[:,1]-REV_division[count]['ymin'])>=-tol2) & ((coord[:,1]-REV_division[count]['ymax'])<=tol2) &
                    ((coord[:,2]-REV_division[count]['zmin'])>=-tol2) & ((coord[:,2]-REV_division[count]['zmax'])<=tol2))[0]
                
                REV_division[count]['indexes']=indexes

                REV_labels[indexes]=count
                print(len(indexes))
                sum=sum+len(indexes)
                count=count+1
                
    if plot_mesh_flag == True:
        plot_mesh(coord,REV_labels,edges,'Capillary Net')

    print('-------------------- COMPUTING INTERSECTIONS --------------------')
    #FIND INTERSECTIONS
    #Compute the intersection between the original mesh and the plane that originates fron the division in REV. New nodes ad edges will be attached to the original lists, using the label '0' for the nodes.

    coord_w_inter=np.copy(coord)
    edges_w_inter=np.copy(edges)
    radii_w_inter=np.copy(radii)
    REV_labels_w_inter=np.copy(REV_labels)


    res=np.where(REV_labels[edges[:,0]]!=REV_labels[edges[:,1]])[0]

    test_edge=edges[res]
    test_radii=radii[res]
    np.shape(test_edge)

    mask = np.ones(len(edges_w_inter), dtype=bool)
    mask[res] = False
    edges_w_inter = edges_w_inter[mask]
    radii_w_inter = radii_w_inter[mask]

    for rr,edge in enumerate(test_edge):
        dist_or=[]
        new_indexes=[]
        
        origin=coord[edge[0]]
        final=coord[edge[1]]
        res_tot=find_intersection2(coord[edge[0]],coord[edge[1]],x_planes,y_planes,z_planes)
        
        for i,res in enumerate(res_tot):
            dist_or.append(np.linalg.norm(origin-res))
            coord_w_inter=np.vstack([coord_w_inter,res])
            REV_labels_w_inter=np.append(REV_labels_w_inter,0)
            ind=np.where((coord_w_inter[:,0]==res[0]) & (coord_w_inter[:,1]==res[1]) & (coord_w_inter[:,2]==res[2]))[0][0]
            new_indexes.append(int(ind))

        dist_or.append(np.linalg.norm(origin-final))
        new_indexes.append(edge[1])
        
        #print(dist_or)
        res=np.argsort(dist_or)
        prev=edge[0]
        for r in res:
            if dist_or[r]!=0:
                edges_w_inter=np.vstack([edges_w_inter,[prev,new_indexes[r]]])
                radii_w_inter=np.append(radii_w_inter,test_radii[rr])
                prev=new_indexes[r]
        

    print(f'Coordinates with intersections: {np.shape(coord_w_inter)}')
    print(f'Edges with intersections: {np.shape(edges_w_inter)}')
    print(f'Labels with intersections: {np.shape(REV_labels_w_inter)}')
    print(f'Radii with intersections: {np.shape(radii_w_inter)}')
    print()
    print('... COMPUTING K VALUES  ...')
    print()

    init_REV=1
    nREV=count-1
    tol=1e-2/8
    tol2=1e-15
    sum=0


    K_m1=np.zeros((count-1,))
    K_m2=np.zeros((count-1,))
    K_x=np.zeros((count-1,))
    K_y=np.zeros((count-1,))
    K_z=np.zeros((count-1,))

    C=np.zeros((count-1,))
    mu_bl_up_x=np.zeros((count-1,))
    mu_bl_up_y=np.zeros((count-1,))
    mu_bl_up_z=np.zeros((count-1,))
    ravg=0
    for dir in [1,2,3]: 
        print(f'------------------------------------------------- DIRECTION: {dir}  -------------------------------------------------')
        for REV in np.arange(init_REV,nREV+1):
            print('-------------------------------------------------------------------------')

            print("REV = "+ str(REV))
            indexes=np.where(((coord_w_inter[:,0]-REV_division[REV]['xmin'])>=-tol2) & ((coord_w_inter[:,0]-REV_division[REV]['xmax'])<=tol2) &
                        ((coord_w_inter[:,1]-REV_division[REV]['ymin'])>=-tol2) & ((coord_w_inter[:,1]-REV_division[REV]['ymax'])<=tol2) &
                        ((coord_w_inter[:,2]-REV_division[REV]['zmin'])>=-tol2) & ((coord_w_inter[:,2]-REV_division[REV]['zmax'])<=tol2))[0]

            #EDGES
            edges_rev=[]
            radii_rev=[]

            index=np.where((REV_labels_w_inter[edges_w_inter[:,0]]==REV_labels_w_inter[edges_w_inter[:,1]]) & (REV_labels_w_inter[edges_w_inter[:,0]]==REV))[0]
            edges_rev_general=edges_w_inter[index]
            radii_rev_general=radii_w_inter[index]

            for rr,edge in enumerate(edges_rev_general):
                ind1=np.where(edge[0]==indexes)[0][0]
                ind2=np.where(edge[1]==indexes)[0][0]
                edges_rev.append([ind1,ind2])
                radii_rev.append(radii_rev_general[rr])

            index=np.where((REV_labels_w_inter[edges_w_inter[:,0]]==0) & (REV_labels_w_inter[edges_w_inter[:,1]]==REV))[0]
            edges_rev_general1=edges_w_inter[index] 
            radii_rev_general1=radii_w_inter[index]

            for rr,edge in enumerate(edges_rev_general1):
                ind1=np.where(edge[0]==indexes)[0][0]
                ind2=np.where(edge[1]==indexes)[0][0]
                edges_rev.append([ind1,ind2])
                radii_rev.append(radii_rev_general1[rr])

            index=np.where((REV_labels_w_inter[edges_w_inter[:,1]]==0) & (REV_labels_w_inter[edges_w_inter[:,0]]==REV))[0]
            edges_rev_general2=edges_w_inter[index]
            radii_rev_general2=radii_w_inter[index]
            

            for rr,edge in enumerate(edges_rev_general2):
                ind1=np.where(edge[0]==indexes)[0][0]
                ind2=np.where(edge[1]==indexes)[0][0]
                edges_rev.append([ind1,ind2])
                radii_rev.append(radii_rev_general2[rr])

            index=np.where((REV_labels_w_inter[edges_w_inter[:,1]]==0) & (REV_labels_w_inter[edges_w_inter[:,0]]==0))[0]
            edges_rev_general3=edges_w_inter[index]
            radii_rev_general3=radii_w_inter[index]


            for rr,edge in enumerate(edges_rev_general3):
                ind1=np.where(edge[0]==indexes)[0]
                ind2=np.where(edge[1]==indexes)[0]

                if len(ind1)!=0 and len(ind2)!=0:
                    edges_rev.append([ind1[0],ind2[0]])
                    radii_rev.append(radii_rev_general3[rr])


            edges_rev=np.array(edges_rev)
            radii_rev=np.array(radii_rev)
            print(np.shape(edges_rev))
            print(np.shape(radii_rev))

            #VERTICES
            coord_rev=np.array(coord_w_inter[indexes])

            #LABELS
            labels_rev=555*np.ones((np.shape(coord_rev)[0],))

            ind1_in=np.where(np.abs(coord_rev[:,0]-REV_division[REV]['xmin'])<tol)[0]
            labels_rev[ind1_in]=111

            ind2_in=np.where(np.abs(coord_rev[:,1]-REV_division[REV]['ymin'])<tol)[0]
            labels_rev[ind2_in]=112

            ind3_in=np.where(np.abs(coord_rev[:,2]-REV_division[REV]['zmin'])<tol)[0]
            labels_rev[ind3_in]=113

            #print(np.shape(ind1_in),np.shape(ind2_in),np.shape(ind3_in))

            ind1_out=np.where(np.abs(coord_rev[:,0]-REV_division[REV]['xmax'])<tol)[0]
            labels_rev[ind1_out]=991

            ind2_out=np.where(np.abs(coord_rev[:,1]-REV_division[REV]['ymax'])<tol)[0]
            labels_rev[ind2_out]=992

            ind3_out=np.where(np.abs(coord_rev[:,2]-REV_division[REV]['zmax'])<tol)[0]
            labels_rev[ind3_out]=993


            #PROCESSING EDGES
            ind_switch=np.where(((labels_rev[edges_rev[:,1]])==555) & ((labels_rev[edges_rev[:,0]])!=(labels_rev[edges_rev[:,1]])))[0]
            temp=np.copy(edges_rev[ind_switch,0])
            edges_rev[ind_switch,0]=edges_rev[ind_switch,1]
            edges_rev[ind_switch,1]=temp

            for ii in [111,112,113,991,992,993]:
                ind=np.where(((labels_rev[edges_rev[:,1]])==ii) & ((labels_rev[edges_rev[:,0]])==(labels_rev[edges_rev[:,1]])))[0]
                mask = np.ones(len(edges_rev), dtype=bool)
                mask[ind] = False
                edges_rev = edges_rev[mask]
                radii_rev = radii_rev[mask]


            G,connected_components=compute_connected_components(edges_rev)
 
            comp1 = max(connected_components, key=len)

            nodes_comp1=np.array(G.subgraph(comp1).nodes())
            temp_edges=np.array(G.subgraph(comp1).edges())
            #print(edges_rev)

            #ADATTARE GLI EDGES AI NUOVI NODI
            coord_comp1=coord_rev[nodes_comp1]
            labels_comp1=labels_rev[nodes_comp1]
            edges_comp1=[]
            radii_comp1=[]
            for edge in temp_edges:
                ind1=np.where((coord_comp1[:,0]==coord_rev[edge[0],0]) &(coord_comp1[:,1]==coord_rev[edge[0],1]) & (coord_comp1[:,2]==coord_rev[edge[0],2]) )[0]
                ind2=np.where((coord_comp1[:,0]==coord_rev[edge[1],0]) &(coord_comp1[:,1]==coord_rev[edge[1],1]) & (coord_comp1[:,2]==coord_rev[edge[1],2]) )[0]
                
                edges_comp1.append([ind1[0],ind2[0]])
                ind=np.where((edges_rev[:,0]==edge[0]) & (edges_rev[:,1]==edge[1]))[0]
                if len(ind)!=0:
                    radii_comp1.append(radii_rev[ind][0])
                else:
                    np.random.seed(42)
                    min_val=np.min(radii_w_inter)
                    max_val=np.max(radii_w_inter)
                    value = np.random.uniform(min_val, max_val)
                    radii_comp1.append(value)
            # print(np.shape(radii_comp1))
            # print(np.shape(edges_comp1))
            #print(radii_comp1)

            edges_comp1=np.array(edges_comp1)
            radii_comp1=np.array(radii_comp1)

            #PROCESSING EDGES
            ind_switch=np.where(((labels_comp1[edges_comp1[:,1]])==555) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            for ii in [111,112,113,991,992,993]:
                ind=np.where(((labels_comp1[edges_comp1[:,1]])==ii) & ((labels_comp1[edges_comp1[:,0]])==(labels_comp1[edges_comp1[:,1]])))[0]
                mask = np.ones(len(edges_comp1), dtype=bool)
                mask[ind] = False
                edges_comp1 = edges_comp1[mask]
                radii_comp1 = radii_comp1[mask]

            #PROCESSING EDGES
            ind_switch=np.where(((labels_comp1[edges_comp1[:,0]])==112) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            #PROCESSING EDGES

            if dir==1:
                indx_in=111
                indx_out=991

            if dir==2:
                indx_in=112
                indx_out=992

            if dir==3:
                indx_in=113
                indx_out=993

            ind_switch=np.where(((labels_comp1[edges_comp1[:,0]])==indx_in) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            ind_switch=np.where(((labels_comp1[edges_comp1[:,0]])==indx_out) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            ind_change=np.where(((labels_comp1[edges_comp1[:,1]])==indx_in) & ((labels_comp1[edges_comp1[:,0]]!=555)))[0]
            for temp_ed in edges_comp1[ind_change]:
                i=temp_ed[0]
                ind=np.where((edges_comp1[:,0]==i) | (edges_comp1[:,1]==i))[0]
                if len(ind)!=1:
                    labels_comp1[i]=555

            ind_change=np.where(((labels_comp1[edges_comp1[:,1]])==indx_out) & ((labels_comp1[edges_comp1[:,0]]!=555)))[0]
            for temp_ed in edges_comp1[ind_change]:
                i=temp_ed[0]
                ind=np.where((edges_comp1[:,0]==i) | (edges_comp1[:,1]==i))[0]
                if len(ind)!=1:
                    labels_comp1[i]=555


            labels_comp1_total=np.copy(labels_comp1)

            ind_out_total=np.where((labels_comp1==991) | (labels_comp1==992) | (labels_comp1==993))[0]
            labels_comp1_total[ind_out_total]=999

            ind_in_total=np.where((labels_comp1==111) | (labels_comp1==112) | (labels_comp1==113))[0]
            labels_comp1_total[ind_in_total]=111

            print('-------------------- SUBMESH INFO --------------------')
            print(f'Coordinates: {np.shape(coord_comp1)}')
            print(f'Edges: {np.shape(edges_comp1)}')
            print(f'Labels: {np.shape(labels_comp1_total)}')
            print(f'Inlet points submesh: {len(ind_in_total)}')
            print(f'Outlet points submesh: {len(ind_out_total)}')
            

            create_mesh_REV(f"{dir_results}/Ncomp_conn_REV_"+ name_mesh +str(REV),coord_comp1,edges_comp1)


            points_comp1,indexes_comp1,Fin_comp1,Fout_comp1=compute_faces(labels_comp1_total,coord_comp1,REV_division[REV]['xmin'],REV_division[REV]['xmax'],REV_division[REV]['ymin'],REV_division[REV]['ymax'],REV_division[REV]['zmin'],REV_division[REV]['zmax'])

            meshQ2 = Mesh()
                
            with XDMFFile(f"{dir_results}/Ncomp_conn_REV_"+ name_mesh +str(REV)+".xdmf") as infile:
                infile.read(meshQ2)

            if dir==1:
                dir_str='x'

            if dir==2:
                dir_str='y'

            if dir==3:
                dir_str='z'

            print(dir,dir_str)
        

            p_tot=compute_pressure2(dir,meshQ2,edges_comp1,coord_comp1,labels_comp1_total, radii_comp1, indexes_comp1['inlet']['total'],indexes_comp1['inlet'][dir_str],indexes_comp1['outlet']['total'],indexes_comp1['outlet'][dir_str],Fin_comp1,Fout_comp1,p_in,p_out)

        

            p_tot_lap=compute_pressure(meshQ2,points_comp1['inlet'][dir_str],points_comp1['outlet'][dir_str],p_in,p_out)
            sol_val=p_tot_lap.compute_vertex_values(meshQ2)

            # diff_p=sol_val-p_tot
            # ind=np.where(diff_p>1e-10)[0]
            # print(diff_p[ind])
            # print(p_tot[ind])
            # print(sol_val[ind])

            k_tot, mu_bl_up=compute_k(dir,meshQ2,radii_comp1,indexes_comp1['outlet'][dir_str],p_tot,coord_comp1,edges_comp1,p_in,p_out)

            k_sol_val, mu_bl_up=compute_k(dir,meshQ2,radii_comp1,indexes_comp1['outlet'][dir_str],sol_val,coord_comp1,edges_comp1,p_in,p_out)

            C_j=compute_C_value(meshQ2,edges_comp1,coord_comp1,radii_comp1,REV_division[REV]['xmax'],REV_division[REV]['xmin'],REV_division[REV]['ymax'],REV_division[REV]['ymin'],REV_division[REV]['zmax'],REV_division[REV]['zmin'])

            K_m1[REV-1]=np.abs(k_tot)
            K_m2[REV-1]=np.abs(k_sol_val)

            if dir==1:
                K_x[REV-1]=np.abs(k_tot)
                mu_bl_up_x[REV-1]=np.abs(mu_bl_up)

            if dir==2:
                K_y[REV-1]=np.abs(k_tot)
                mu_bl_up_y[REV-1]=np.abs(mu_bl_up)

            if dir==3:
                K_z[REV-1]=np.abs(k_tot)
                mu_bl_up_z[REV-1]=np.abs(mu_bl_up)


            C[REV-1]=C_j

            print(k_tot)
            print(k_sol_val)
            print(C_j)

            if M_plot_flag==True:
                plot_mesh(coord_comp1,p_tot,edges_comp1,' Matrix Pressure REV '+str(REV))
            
            if lap_plot_flag==True:
                plot_mesh(coord_comp1,sol_val,edges_comp1,' Lap Pressure REV '+str(REV))
            mu_bl_up=mu_bl_up_x

    return K_x,K_y,K_z,C,mu_bl_up
    #return K_x,K_y,K_z,C,mu_bl_up_x,mu_bl_up_y,mu_bl_up_z
def scaling_analysis_TOTAL(meshQ,radii, Q_markers,n_div_x,n_div_y,n_div_z,p_in,p_out,x,y,z,plot_mesh_flag=False, M_plot_flag=False,lap_plot_flag=False):
    #NOTE THE DIVISION IN MADE FOR A CUBE OF DIM (-1,1)^3
    #Mesh Info
    coord=meshQ.coordinates()
    labels=Q_markers.array()
    edges=meshQ.cells()

    print('-------------------- MESH INFO --------------------')
    print(f'Coordinates: {np.shape(coord)}')
    print(f'Edges: {np.shape(edges)}')
    print(f'Radii: {np.shape(radii)}')
    print(f'Labels: {np.shape(labels)}')
    print(f'Interior Points: {np.shape(np.where(labels==555)[0])}')
    print(f'Inlet Points: {np.shape(np.where(labels==111)[0])}')
    print(f'Outlet Points: {np.shape(np.where(labels==999)[0])}')

    print()
    
    if plot_mesh_flag == True:
        plot_mesh(coord,labels,edges,'Capillary Net')

    print(f'Total REV number: {n_div_x*n_div_y*n_div_z}')

    #Division
    REV_division={}
    REV_labels=np.zeros(np.shape(labels))
    #np.shape(REV_labels)
    count=1
    sum=0
    tol2=1e-15
    xmin=-1
    xmax=1
    ymin=-1
    ymax=1
    zmin=-1
    zmax=1

    #x= np.linspace(xmin,xmax,n_div_x+1)
    x_planes=x[1:-1]

    #y= np.linspace(ymin,ymax,n_div_y+1)
    y_planes=y[1:-1]

    #z=np.linspace(zmin,zmax,n_div_z+1)
    z_planes=z[1:-1]

    for i in np.arange(len(x)-1):
        for j in np.arange(len(y)-1):
            for k in np.arange(len(z)-1):
                REV_division[count]={'xmin':np.maximum(xmin,x[i])}
                REV_division[count]['xmax']=np.minimum(xmax,x[i+1])
                REV_division[count]['ymin']=np.maximum(ymin,y[j])
                REV_division[count]['ymax']=np.minimum(ymax,y[j+1])
                REV_division[count]['zmin']=np.maximum(zmin,z[k])
                REV_division[count]['zmax']=np.minimum(zmax,z[k+1])

                indexes=np.where(((coord[:,0]-REV_division[count]['xmin'])>=-tol2) & ((coord[:,0]-REV_division[count]['xmax'])<=tol2) &
                    ((coord[:,1]-REV_division[count]['ymin'])>=-tol2) & ((coord[:,1]-REV_division[count]['ymax'])<=tol2) &
                    ((coord[:,2]-REV_division[count]['zmin'])>=-tol2) & ((coord[:,2]-REV_division[count]['zmax'])<=tol2))[0]
                
                REV_division[count]['indexes']=indexes

                REV_labels[indexes]=count
                print(len(indexes))
                sum=sum+len(indexes)
                count=count+1
                
    if plot_mesh_flag == True:
        plot_mesh(coord,REV_labels,edges,'Capillary Net')

    print('-------------------- COMPUTING INTERSECTIONS --------------------')
    #FIND INTERSECTIONS
    coord_w_inter=np.copy(coord)
    edges_w_inter=np.copy(edges)
    radii_w_inter=np.copy(radii)
    REV_labels_w_inter=np.copy(REV_labels)


    res=np.where(REV_labels[edges[:,0]]!=REV_labels[edges[:,1]])[0]

    test_edge=edges[res]
    test_radii=radii[res]
    np.shape(test_edge)

    mask = np.ones(len(edges_w_inter), dtype=bool)
    mask[res] = False
    edges_w_inter = edges_w_inter[mask]
    radii_w_inter = radii_w_inter[mask]

    for rr,edge in enumerate(test_edge):
        dist_or=[]
        new_indexes=[]
        
        origin=coord[edge[0]]
        final=coord[edge[1]]
        res_tot=find_intersection2(coord[edge[0]],coord[edge[1]],x_planes,y_planes,z_planes)
        
        for i,res in enumerate(res_tot):
            dist_or.append(np.linalg.norm(origin-res))
            coord_w_inter=np.vstack([coord_w_inter,res])
            REV_labels_w_inter=np.append(REV_labels_w_inter,0)
            ind=np.where((coord_w_inter[:,0]==res[0]) & (coord_w_inter[:,1]==res[1]) & (coord_w_inter[:,2]==res[2]))[0][0]
            new_indexes.append(int(ind))

        dist_or.append(np.linalg.norm(origin-final))
        new_indexes.append(edge[1])
        
        #print(dist_or)
        res=np.argsort(dist_or)
        prev=edge[0]
        for r in res:
            if dist_or[r]!=0:
                edges_w_inter=np.vstack([edges_w_inter,[prev,new_indexes[r]]])
                radii_w_inter=np.append(radii_w_inter,test_radii[rr])
                prev=new_indexes[r]
        

    print(f'Coordinates with intersections: {np.shape(coord_w_inter)}')
    print(f'Edges with intersections: {np.shape(edges_w_inter)}')
    print(f'Labels with intersections: {np.shape(REV_labels_w_inter)}')
    print(f'Radii with intersections: {np.shape(radii_w_inter)}')
    print()
    print('... COMPUTING K VALUES  ...')
    print()

    init_REV=1
    nREV=count-1
    tol=1e-2/8
    tol2=1e-15
    sum=0


    R_IN=[]
    R_OUT=[]
    RTOT=[]
    
    
    ravg=0
    for dir in [1,2,3]: 
        print(f'------------------------------------------------- DIRECTION: {dir}  -------------------------------------------------')
        for REV in np.arange(init_REV,nREV+1):
            print('-------------------------------------------------------------------------')

            print("REV = "+ str(REV))
            indexes=np.where(((coord_w_inter[:,0]-REV_division[REV]['xmin'])>=-tol2) & ((coord_w_inter[:,0]-REV_division[REV]['xmax'])<=tol2) &
                        ((coord_w_inter[:,1]-REV_division[REV]['ymin'])>=-tol2) & ((coord_w_inter[:,1]-REV_division[REV]['ymax'])<=tol2) &
                        ((coord_w_inter[:,2]-REV_division[REV]['zmin'])>=-tol2) & ((coord_w_inter[:,2]-REV_division[REV]['zmax'])<=tol2))[0]

            #EDGES
            edges_rev=[]
            radii_rev=[]

            index=np.where((REV_labels_w_inter[edges_w_inter[:,0]]==REV_labels_w_inter[edges_w_inter[:,1]]) & (REV_labels_w_inter[edges_w_inter[:,0]]==REV))[0]
            edges_rev_general=edges_w_inter[index]
            radii_rev_general=radii_w_inter[index]

            for rr,edge in enumerate(edges_rev_general):
                ind1=np.where(edge[0]==indexes)[0][0]
                ind2=np.where(edge[1]==indexes)[0][0]
                edges_rev.append([ind1,ind2])
                radii_rev.append(radii_rev_general[rr])

            index=np.where((REV_labels_w_inter[edges_w_inter[:,0]]==0) & (REV_labels_w_inter[edges_w_inter[:,1]]==REV))[0]
            edges_rev_general1=edges_w_inter[index] 
            radii_rev_general1=radii_w_inter[index]

            for rr,edge in enumerate(edges_rev_general1):
                ind1=np.where(edge[0]==indexes)[0][0]
                ind2=np.where(edge[1]==indexes)[0][0]
                edges_rev.append([ind1,ind2])
                radii_rev.append(radii_rev_general1[rr])

            index=np.where((REV_labels_w_inter[edges_w_inter[:,1]]==0) & (REV_labels_w_inter[edges_w_inter[:,0]]==REV))[0]
            edges_rev_general2=edges_w_inter[index]
            radii_rev_general2=radii_w_inter[index]
            

            for rr,edge in enumerate(edges_rev_general2):
                ind1=np.where(edge[0]==indexes)[0][0]
                ind2=np.where(edge[1]==indexes)[0][0]
                edges_rev.append([ind1,ind2])
                radii_rev.append(radii_rev_general2[rr])

            index=np.where((REV_labels_w_inter[edges_w_inter[:,1]]==0) & (REV_labels_w_inter[edges_w_inter[:,0]]==0))[0]
            edges_rev_general3=edges_w_inter[index]
            radii_rev_general3=radii_w_inter[index]


            for rr,edge in enumerate(edges_rev_general3):
                ind1=np.where(edge[0]==indexes)[0]
                ind2=np.where(edge[1]==indexes)[0]

                if len(ind1)!=0 and len(ind2)!=0:
                    edges_rev.append([ind1[0],ind2[0]])
                    radii_rev.append(radii_rev_general3[rr])


            edges_rev=np.array(edges_rev)
            radii_rev=np.array(radii_rev)
            print(np.shape(edges_rev))
            print(np.shape(radii_rev))

            #VERTICES
            coord_rev=np.array(coord_w_inter[indexes])

            #LABELS
            labels_rev=555*np.ones((np.shape(coord_rev)[0],))

            ind1_in=np.where(np.abs(coord_rev[:,0]-REV_division[REV]['xmin'])<tol)[0]
            labels_rev[ind1_in]=111

            ind2_in=np.where(np.abs(coord_rev[:,1]-REV_division[REV]['ymin'])<tol)[0]
            labels_rev[ind2_in]=112

            ind3_in=np.where(np.abs(coord_rev[:,2]-REV_division[REV]['zmin'])<tol)[0]
            labels_rev[ind3_in]=113

            #print(np.shape(ind1_in),np.shape(ind2_in),np.shape(ind3_in))

            ind1_out=np.where(np.abs(coord_rev[:,0]-REV_division[REV]['xmax'])<tol)[0]
            labels_rev[ind1_out]=991

            ind2_out=np.where(np.abs(coord_rev[:,1]-REV_division[REV]['ymax'])<tol)[0]
            labels_rev[ind2_out]=992

            ind3_out=np.where(np.abs(coord_rev[:,2]-REV_division[REV]['zmax'])<tol)[0]
            labels_rev[ind3_out]=993


            #PROCESSING EDGES
            ind_switch=np.where(((labels_rev[edges_rev[:,1]])==555) & ((labels_rev[edges_rev[:,0]])!=(labels_rev[edges_rev[:,1]])))[0]
            temp=np.copy(edges_rev[ind_switch,0])
            edges_rev[ind_switch,0]=edges_rev[ind_switch,1]
            edges_rev[ind_switch,1]=temp

            for ii in [111,112,113,991,992,993]:
                ind=np.where(((labels_rev[edges_rev[:,1]])==ii) & ((labels_rev[edges_rev[:,0]])==(labels_rev[edges_rev[:,1]])))[0]
                mask = np.ones(len(edges_rev), dtype=bool)
                mask[ind] = False
                edges_rev = edges_rev[mask]
                radii_rev = radii_rev[mask]


            G,connected_components=compute_connected_components(edges_rev)
 
            comp1 = max(connected_components, key=len)

            nodes_comp1=np.array(G.subgraph(comp1).nodes())
            temp_edges=np.array(G.subgraph(comp1).edges())
            #print(edges_rev)

            #ADATTARE GLI EDGES AI NUOVI NODI
            coord_comp1=coord_rev[nodes_comp1]
            labels_comp1=labels_rev[nodes_comp1]
            edges_comp1=[]
            radii_comp1=[]
            for edge in temp_edges:
                ind1=np.where((coord_comp1[:,0]==coord_rev[edge[0],0]) &(coord_comp1[:,1]==coord_rev[edge[0],1]) & (coord_comp1[:,2]==coord_rev[edge[0],2]) )[0]
                ind2=np.where((coord_comp1[:,0]==coord_rev[edge[1],0]) &(coord_comp1[:,1]==coord_rev[edge[1],1]) & (coord_comp1[:,2]==coord_rev[edge[1],2]) )[0]
                
                edges_comp1.append([ind1[0],ind2[0]])
                ind=np.where((edges_rev[:,0]==edge[0]) & (edges_rev[:,1]==edge[1]))[0]
                if len(ind)!=0:
                    radii_comp1.append(radii_rev[ind][0])
                else:
                    np.random.seed(42)
                    min_val=np.min(radii_w_inter)
                    max_val=np.max(radii_w_inter)
                    value = np.random.uniform(min_val, max_val)
                    radii_comp1.append(value)
            # print(np.shape(radii_comp1))
            # print(np.shape(edges_comp1))
            #print(radii_comp1)

            edges_comp1=np.array(edges_comp1)
            radii_comp1=np.array(radii_comp1)

            #PROCESSING EDGES
            ind_switch=np.where(((labels_comp1[edges_comp1[:,1]])==555) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            for ii in [111,112,113,991,992,993]:
                ind=np.where(((labels_comp1[edges_comp1[:,1]])==ii) & ((labels_comp1[edges_comp1[:,0]])==(labels_comp1[edges_comp1[:,1]])))[0]
                mask = np.ones(len(edges_comp1), dtype=bool)
                mask[ind] = False
                edges_comp1 = edges_comp1[mask]
                radii_comp1 = radii_comp1[mask]

            #PROCESSING EDGES
            ind_switch=np.where(((labels_comp1[edges_comp1[:,0]])==112) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            #PROCESSING EDGES

            if dir==1:
                indx_in=111
                indx_out=991

            if dir==2:
                indx_in=112
                indx_out=992

            if dir==3:
                indx_in=113
                indx_out=993

            ind_switch=np.where(((labels_comp1[edges_comp1[:,0]])==indx_in) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            ind_switch=np.where(((labels_comp1[edges_comp1[:,0]])==indx_out) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            ind_change=np.where(((labels_comp1[edges_comp1[:,1]])==indx_in) & ((labels_comp1[edges_comp1[:,0]]!=555)))[0]
            for temp_ed in edges_comp1[ind_change]:
                i=temp_ed[0]
                ind=np.where((edges_comp1[:,0]==i) | (edges_comp1[:,1]==i))[0]
                if len(ind)!=1:
                    labels_comp1[i]=555

            ind_change=np.where(((labels_comp1[edges_comp1[:,1]])==indx_out) & ((labels_comp1[edges_comp1[:,0]]!=555)))[0]
            for temp_ed in edges_comp1[ind_change]:
                i=temp_ed[0]
                ind=np.where((edges_comp1[:,0]==i) | (edges_comp1[:,1]==i))[0]
                if len(ind)!=1:
                    labels_comp1[i]=555


            labels_comp1_total=np.copy(labels_comp1)

            ind_out_total=np.where((labels_comp1==991) | (labels_comp1==992) | (labels_comp1==993))[0]
            labels_comp1_total[ind_out_total]=999

            ind_in_total=np.where((labels_comp1==111) | (labels_comp1==112) | (labels_comp1==113))[0]
            labels_comp1_total[ind_in_total]=111

            print('-------------------- SUBMESH INFO --------------------')
            print(f'Coordinates: {np.shape(coord_comp1)}')
            print(f'Edges: {np.shape(edges_comp1)}')
            print(f'Labels: {np.shape(labels_comp1_total)}')
            print(f'Inlet points submesh: {len(ind_in_total)}')
            print(f'Outlet points submesh: {len(ind_out_total)}')
            
            rin=len(ind_in_total)/len(coord_comp1)
            print(f'RATIO IN: {rin}')
            rout=len(ind_out_total)/len(coord_comp1)
            print(f'RATIO IN: {rout}')

            rTot=(len(ind_out_total)+len(ind_in_total))/len(coord_comp1)
            print(f'RATIO TOT: {rTot}')
            ravg=ravg+rTot
            R_IN.append(rin)
            R_OUT.append(rout)
            RTOT.append(rTot)

            
    RAVG=ravg/(nREV*3)
    print(f'RATIO AVG: {ravg/(nREV*3)}')
    return R_IN,R_OUT,RTOT,RAVG

def scaling_analysis(meshQ,radii, Q_markers,n_div_x,n_div_y,n_div_z,plot_mesh_flag=False):
    #Gives the ration between the inlet/outlet points and the total points after the division, to conduct a very simplified scaling analysis.

    #NOTE THE DIVISION IN MADE FOR A CUBE OF DIM (-1,1)^3
    #Mesh Info
    coord=meshQ.coordinates()
    labels=Q_markers.array()
    edges=meshQ.cells()

    print('-------------------- MESH INFO --------------------')
    print(f'Coordinates: {np.shape(coord)}')
    print(f'Edges: {np.shape(edges)}')
    print(f'Radii: {np.shape(radii)}')
    print(f'Labels: {np.shape(labels)}')
    print(f'Interior Points: {np.shape(np.where(labels==555)[0])}')
    print(f'Inlet Points: {np.shape(np.where(labels==111)[0])}')
    print(f'Outlet Points: {np.shape(np.where(labels==999)[0])}')

    print()
    
    if plot_mesh_flag == True:
        plot_mesh(coord,labels,edges,'Capillary Net')

    print(f'Total REV number: {n_div_x*n_div_y*n_div_z}')

    #Division
    REV_division={}
    REV_labels=np.zeros(np.shape(labels))
    #np.shape(REV_labels)
    count=1
    sum=0
    tol2=1e-15
    xmin=-1
    xmax=1
    ymin=-1
    ymax=1
    zmin=-1
    zmax=1

    x= np.linspace(xmin,xmax,n_div_x+1)
    x_planes=x[1:-1]

    y= np.linspace(ymin,ymax,n_div_y+1)
    y_planes=y[1:-1]

    z=np.linspace(zmin,zmax,n_div_z+1)
    z_planes=z[1:-1]

    for i in np.arange(len(x)-1):
        for j in np.arange(len(y)-1):
            for k in np.arange(len(z)-1):
                REV_division[count]={'xmin':np.maximum(xmin,x[i])}
                REV_division[count]['xmax']=np.minimum(xmax,x[i+1])
                REV_division[count]['ymin']=np.maximum(ymin,y[j])
                REV_division[count]['ymax']=np.minimum(ymax,y[j+1])
                REV_division[count]['zmin']=np.maximum(zmin,z[k])
                REV_division[count]['zmax']=np.minimum(zmax,z[k+1])

                indexes=np.where(((coord[:,0]-REV_division[count]['xmin'])>=-tol2) & ((coord[:,0]-REV_division[count]['xmax'])<=tol2) &
                    ((coord[:,1]-REV_division[count]['ymin'])>=-tol2) & ((coord[:,1]-REV_division[count]['ymax'])<=tol2) &
                    ((coord[:,2]-REV_division[count]['zmin'])>=-tol2) & ((coord[:,2]-REV_division[count]['zmax'])<=tol2))[0]
                
                REV_division[count]['indexes']=indexes

                REV_labels[indexes]=count
                print(len(indexes))
                sum=sum+len(indexes)
                count=count+1
                
    if plot_mesh_flag == True:
        plot_mesh(coord,REV_labels,edges,'Capillary Net')

    print('-------------------- COMPUTING INTERSECTIONS --------------------')
    #FIND INTERSECTIONS
    coord_w_inter=np.copy(coord)
    edges_w_inter=np.copy(edges)
    radii_w_inter=np.copy(radii)
    REV_labels_w_inter=np.copy(REV_labels)


    res=np.where(REV_labels[edges[:,0]]!=REV_labels[edges[:,1]])[0]

    test_edge=edges[res]
    test_radii=radii[res]
    np.shape(test_edge)

    mask = np.ones(len(edges_w_inter), dtype=bool)
    mask[res] = False
    edges_w_inter = edges_w_inter[mask]
    radii_w_inter = radii_w_inter[mask]

    for rr,edge in enumerate(test_edge):
        dist_or=[]
        new_indexes=[]
        
        origin=coord[edge[0]]
        final=coord[edge[1]]
        res_tot=find_intersection2(coord[edge[0]],coord[edge[1]],x_planes,y_planes,z_planes)
        
        for i,res in enumerate(res_tot):
            dist_or.append(np.linalg.norm(origin-res))
            coord_w_inter=np.vstack([coord_w_inter,res])
            REV_labels_w_inter=np.append(REV_labels_w_inter,0)
            ind=np.where((coord_w_inter[:,0]==res[0]) & (coord_w_inter[:,1]==res[1]) & (coord_w_inter[:,2]==res[2]))[0][0]
            new_indexes.append(int(ind))

        dist_or.append(np.linalg.norm(origin-final))
        new_indexes.append(edge[1])
        
        #print(dist_or)
        res=np.argsort(dist_or)
        prev=edge[0]
        for r in res:
            if dist_or[r]!=0:
                edges_w_inter=np.vstack([edges_w_inter,[prev,new_indexes[r]]])
                radii_w_inter=np.append(radii_w_inter,test_radii[rr])
                prev=new_indexes[r]
        

    print(f'Coordinates with intersections: {np.shape(coord_w_inter)}')
    print(f'Edges with intersections: {np.shape(edges_w_inter)}')
    print(f'Labels with intersections: {np.shape(REV_labels_w_inter)}')
    print(f'Radii with intersections: {np.shape(radii_w_inter)}')
    print()
    print('... COMPUTING K VALUES  ...')
    print()

    init_REV=1
    nREV=count-1
    tol=1e-2/8
    tol2=1e-15
    sum=0


    R_IN=[]
    R_OUT=[]
    RTOT=[]
    
    
    ravg=0
    for dir in [1,2,3]: 
        print(f'------------------------------------------------- DIRECTION: {dir}  -------------------------------------------------')
        for REV in np.arange(init_REV,nREV+1):
            print('-------------------------------------------------------------------------')

            print("REV = "+ str(REV))
            indexes=np.where(((coord_w_inter[:,0]-REV_division[REV]['xmin'])>=-tol2) & ((coord_w_inter[:,0]-REV_division[REV]['xmax'])<=tol2) &
                        ((coord_w_inter[:,1]-REV_division[REV]['ymin'])>=-tol2) & ((coord_w_inter[:,1]-REV_division[REV]['ymax'])<=tol2) &
                        ((coord_w_inter[:,2]-REV_division[REV]['zmin'])>=-tol2) & ((coord_w_inter[:,2]-REV_division[REV]['zmax'])<=tol2))[0]

            #EDGES
            edges_rev=[]
            radii_rev=[]

            index=np.where((REV_labels_w_inter[edges_w_inter[:,0]]==REV_labels_w_inter[edges_w_inter[:,1]]) & (REV_labels_w_inter[edges_w_inter[:,0]]==REV))[0]
            edges_rev_general=edges_w_inter[index]
            radii_rev_general=radii_w_inter[index]

            for rr,edge in enumerate(edges_rev_general):
                ind1=np.where(edge[0]==indexes)[0][0]
                ind2=np.where(edge[1]==indexes)[0][0]
                edges_rev.append([ind1,ind2])
                radii_rev.append(radii_rev_general[rr])

            index=np.where((REV_labels_w_inter[edges_w_inter[:,0]]==0) & (REV_labels_w_inter[edges_w_inter[:,1]]==REV))[0]
            edges_rev_general1=edges_w_inter[index] 
            radii_rev_general1=radii_w_inter[index]

            for rr,edge in enumerate(edges_rev_general1):
                ind1=np.where(edge[0]==indexes)[0][0]
                ind2=np.where(edge[1]==indexes)[0][0]
                edges_rev.append([ind1,ind2])
                radii_rev.append(radii_rev_general1[rr])

            index=np.where((REV_labels_w_inter[edges_w_inter[:,1]]==0) & (REV_labels_w_inter[edges_w_inter[:,0]]==REV))[0]
            edges_rev_general2=edges_w_inter[index]
            radii_rev_general2=radii_w_inter[index]
            

            for rr,edge in enumerate(edges_rev_general2):
                ind1=np.where(edge[0]==indexes)[0][0]
                ind2=np.where(edge[1]==indexes)[0][0]
                edges_rev.append([ind1,ind2])
                radii_rev.append(radii_rev_general2[rr])

            index=np.where((REV_labels_w_inter[edges_w_inter[:,1]]==0) & (REV_labels_w_inter[edges_w_inter[:,0]]==0))[0]
            edges_rev_general3=edges_w_inter[index]
            radii_rev_general3=radii_w_inter[index]


            for rr,edge in enumerate(edges_rev_general3):
                ind1=np.where(edge[0]==indexes)[0]
                ind2=np.where(edge[1]==indexes)[0]

                if len(ind1)!=0 and len(ind2)!=0:
                    edges_rev.append([ind1[0],ind2[0]])
                    radii_rev.append(radii_rev_general3[rr])


            edges_rev=np.array(edges_rev)
            radii_rev=np.array(radii_rev)
            print(np.shape(edges_rev))
            print(np.shape(radii_rev))

            #VERTICES
            coord_rev=np.array(coord_w_inter[indexes])

            #LABELS
            labels_rev=555*np.ones((np.shape(coord_rev)[0],))

            ind1_in=np.where(np.abs(coord_rev[:,0]-REV_division[REV]['xmin'])<tol)[0]
            labels_rev[ind1_in]=111

            ind2_in=np.where(np.abs(coord_rev[:,1]-REV_division[REV]['ymin'])<tol)[0]
            labels_rev[ind2_in]=112

            ind3_in=np.where(np.abs(coord_rev[:,2]-REV_division[REV]['zmin'])<tol)[0]
            labels_rev[ind3_in]=113

            #print(np.shape(ind1_in),np.shape(ind2_in),np.shape(ind3_in))

            ind1_out=np.where(np.abs(coord_rev[:,0]-REV_division[REV]['xmax'])<tol)[0]
            labels_rev[ind1_out]=991

            ind2_out=np.where(np.abs(coord_rev[:,1]-REV_division[REV]['ymax'])<tol)[0]
            labels_rev[ind2_out]=992

            ind3_out=np.where(np.abs(coord_rev[:,2]-REV_division[REV]['zmax'])<tol)[0]
            labels_rev[ind3_out]=993


            #PROCESSING EDGES
            ind_switch=np.where(((labels_rev[edges_rev[:,1]])==555) & ((labels_rev[edges_rev[:,0]])!=(labels_rev[edges_rev[:,1]])))[0]
            temp=np.copy(edges_rev[ind_switch,0])
            edges_rev[ind_switch,0]=edges_rev[ind_switch,1]
            edges_rev[ind_switch,1]=temp

            for ii in [111,112,113,991,992,993]:
                ind=np.where(((labels_rev[edges_rev[:,1]])==ii) & ((labels_rev[edges_rev[:,0]])==(labels_rev[edges_rev[:,1]])))[0]
                mask = np.ones(len(edges_rev), dtype=bool)
                mask[ind] = False
                edges_rev = edges_rev[mask]
                radii_rev = radii_rev[mask]


            G,connected_components=compute_connected_components(edges_rev)
 
            comp1 = max(connected_components, key=len)

            nodes_comp1=np.array(G.subgraph(comp1).nodes())
            temp_edges=np.array(G.subgraph(comp1).edges())
            #print(edges_rev)

            #ADATTARE GLI EDGES AI NUOVI NODI
            coord_comp1=coord_rev[nodes_comp1]
            labels_comp1=labels_rev[nodes_comp1]
            edges_comp1=[]
            radii_comp1=[]
            for edge in temp_edges:
                ind1=np.where((coord_comp1[:,0]==coord_rev[edge[0],0]) &(coord_comp1[:,1]==coord_rev[edge[0],1]) & (coord_comp1[:,2]==coord_rev[edge[0],2]) )[0]
                ind2=np.where((coord_comp1[:,0]==coord_rev[edge[1],0]) &(coord_comp1[:,1]==coord_rev[edge[1],1]) & (coord_comp1[:,2]==coord_rev[edge[1],2]) )[0]
                
                edges_comp1.append([ind1[0],ind2[0]])
                ind=np.where((edges_rev[:,0]==edge[0]) & (edges_rev[:,1]==edge[1]))[0]
                if len(ind)!=0:
                    radii_comp1.append(radii_rev[ind][0])
                else:
                    np.random.seed(42)
                    min_val=np.min(radii_w_inter)
                    max_val=np.max(radii_w_inter)
                    value = np.random.uniform(min_val, max_val)
                    radii_comp1.append(value)
            # print(np.shape(radii_comp1))
            # print(np.shape(edges_comp1))
            #print(radii_comp1)

            edges_comp1=np.array(edges_comp1)
            radii_comp1=np.array(radii_comp1)

            #PROCESSING EDGES
            ind_switch=np.where(((labels_comp1[edges_comp1[:,1]])==555) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            for ii in [111,112,113,991,992,993]:
                ind=np.where(((labels_comp1[edges_comp1[:,1]])==ii) & ((labels_comp1[edges_comp1[:,0]])==(labels_comp1[edges_comp1[:,1]])))[0]
                mask = np.ones(len(edges_comp1), dtype=bool)
                mask[ind] = False
                edges_comp1 = edges_comp1[mask]
                radii_comp1 = radii_comp1[mask]

            #PROCESSING EDGES
            ind_switch=np.where(((labels_comp1[edges_comp1[:,0]])==112) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            #PROCESSING EDGES

            if dir==1:
                indx_in=111
                indx_out=991

            if dir==2:
                indx_in=112
                indx_out=992

            if dir==3:
                indx_in=113
                indx_out=993

            ind_switch=np.where(((labels_comp1[edges_comp1[:,0]])==indx_in) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            ind_switch=np.where(((labels_comp1[edges_comp1[:,0]])==indx_out) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            ind_change=np.where(((labels_comp1[edges_comp1[:,1]])==indx_in) & ((labels_comp1[edges_comp1[:,0]]!=555)))[0]
            for temp_ed in edges_comp1[ind_change]:
                i=temp_ed[0]
                ind=np.where((edges_comp1[:,0]==i) | (edges_comp1[:,1]==i))[0]
                if len(ind)!=1:
                    labels_comp1[i]=555

            ind_change=np.where(((labels_comp1[edges_comp1[:,1]])==indx_out) & ((labels_comp1[edges_comp1[:,0]]!=555)))[0]
            for temp_ed in edges_comp1[ind_change]:
                i=temp_ed[0]
                ind=np.where((edges_comp1[:,0]==i) | (edges_comp1[:,1]==i))[0]
                if len(ind)!=1:
                    labels_comp1[i]=555


            labels_comp1_total=np.copy(labels_comp1)

            ind_out_total=np.where((labels_comp1==991) | (labels_comp1==992) | (labels_comp1==993))[0]
            labels_comp1_total[ind_out_total]=999

            ind_in_total=np.where((labels_comp1==111) | (labels_comp1==112) | (labels_comp1==113))[0]
            labels_comp1_total[ind_in_total]=111

            print('-------------------- SUBMESH INFO --------------------')
            print(f'Coordinates: {np.shape(coord_comp1)}')
            print(f'Edges: {np.shape(edges_comp1)}')
            print(f'Labels: {np.shape(labels_comp1_total)}')
            print(f'Inlet points submesh: {len(ind_in_total)}')
            print(f'Outlet points submesh: {len(ind_out_total)}')
            
            rin=len(ind_in_total)/len(coord_comp1)
            print(f'RATIO IN: {rin}')
            rout=len(ind_out_total)/len(coord_comp1)
            print(f'RATIO IN: {rout}')

            rTot=(len(ind_out_total)+len(ind_in_total))/len(coord_comp1)
            print(f'RATIO TOT: {rTot}')
            ravg=ravg+rTot
            R_IN.append(rin)
            R_OUT.append(rout)
            RTOT.append(rTot)

            
    RAVG=ravg/(nREV*3)
    print(f'RATIO AVG: {ravg/(nREV*3)}')
    return R_IN,R_OUT,RTOT,RAVG

def compute_faces_TOTAL(labels,labels_comp1,coord):
    ''' Function the associates the face labels to the respective inlet and outlet points of a mesh that is not cubic.
        INPUT: 
            labels : labels of the vertices of the network.
            labels_comp1 : labels of the vertices of the network that take into account the direction.
            coord :  coordinates of the vertices/nodes of the network.

        OUTPUT:
            points : dictionary that contains the inlet/outlet points grouped by their belonging to each one of the faces.
            indexes : dictionary that contains the inlet/outlet indexes grouped by their belonging to each one of the faces.
            F_in : labels associated with each inlet points identifyng the face of belonging. (Ex: the label '1' is associated the the inlet point on the inlet face in the x direction.)
            F_out : labels associated with each outlet points identifyng the face of belonging.
    '''
    points={}
    indexes={}

    outlet_indexes=np.where(labels==999)[0]
    inlet_indexes=np.where(labels==111)[0]
    outlet_points=coord[outlet_indexes]
    inlet_points=coord[inlet_indexes]

    tol=1e-2/8

    F_out=np.zeros((np.shape(outlet_points)[0],1))

    ind1_out=np.where(labels_comp1==991)[0]
    for ii in ind1_out:
        check=np.where((coord[ii,0]==outlet_points[:,0]) & (coord[ii,1]==outlet_points[:,1]) & (coord[ii,2]==outlet_points[:,2]))[0]
        F_out[check]=1

    ind2_out=np.where(labels_comp1==992)[0]
    for ii in ind2_out:
        check=np.where((coord[ii,0]==outlet_points[:,0]) & (coord[ii,1]==outlet_points[:,1]) & (coord[ii,2]==outlet_points[:,2]))[0]
        F_out[check]=2
    #outlet_points_y=outlet_points[ind2_out]
    #print(np.shape(outlet_points_y))
    ind3_out=np.where(labels_comp1==993)[0]
    for ii in ind3_out:
        check=np.where((coord[ii,0]==outlet_points[:,0]) & (coord[ii,1]==outlet_points[:,1]) & (coord[ii,2]==outlet_points[:,2]))[0]
        F_out[check]=3

    outlet_points_x=outlet_points[np.where(F_out==1)[0]]
    outlet_points_y=outlet_points[np.where(F_out==2)[0]]
    outlet_points_z=outlet_points[np.where(F_out==3)[0]]

    outlet_indexes_x=outlet_indexes[np.where(F_out==1)[0]]
    outlet_indexes_y=outlet_indexes[np.where(F_out==2)[0]]
    outlet_indexes_z=outlet_indexes[np.where(F_out==3)[0]]


    #print(np.shape(ind1_out)[0]+np.shape(ind2_out)[0]+np.shape(ind3_out)[0])

    F_in=np.zeros((np.shape(inlet_points)[0],1))

    ind1_in=np.where(labels_comp1==111)[0]
    for ii in ind1_in:
        check=np.where((coord[ii,0]==inlet_points[:,0]) & (coord[ii,1]==inlet_points[:,1]) & (coord[ii,2]==inlet_points[:,2]))[0]
        F_in[check]=1

    ind2_in=np.where(labels_comp1==112)[0]
    for ii in ind2_in:
        check=np.where((coord[ii,0]==inlet_points[:,0]) & (coord[ii,1]==inlet_points[:,1]) & (coord[ii,2]==inlet_points[:,2]))[0]
        F_in[check]=2

    ind3_in=np.where(labels_comp1==113)[0]
    for ii in ind3_in:
        check=np.where((coord[ii,0]==inlet_points[:,0]) & (coord[ii,1]==inlet_points[:,1]) & (coord[ii,2]==inlet_points[:,2]))[0]
        F_in[check]=3

    inlet_points_x=inlet_points[np.where(F_in==1)[0]]
    inlet_points_y=inlet_points[np.where(F_in==2)[0]]
    inlet_points_z=inlet_points[np.where(F_in==3)[0]]

    inlet_indexes_x=inlet_indexes[np.where(F_in==1)[0]]
    inlet_indexes_y=inlet_indexes[np.where(F_in==2)[0]]
    inlet_indexes_z=inlet_indexes[np.where(F_in==3)[0]]

    points['inlet']={'x': inlet_points_x}
    points['inlet']['y']=inlet_points_y
    points['inlet']['z']=inlet_points_z
    points['inlet']['total']=inlet_points


    points['outlet']={'x': outlet_points_x}
    points['outlet']['y']=outlet_points_y
    points['outlet']['z']=outlet_points_z
    points['outlet']['total']=outlet_points


    indexes['inlet']={'x': inlet_indexes_x}
    indexes['inlet']['y']=inlet_indexes_y
    indexes['inlet']['z']=inlet_indexes_z
    indexes['inlet']['total']=inlet_indexes


    indexes['outlet']={'x': outlet_indexes_x}
    indexes['outlet']['y']=outlet_indexes_y
    indexes['outlet']['z']=outlet_indexes_z
    indexes['outlet']['total']=outlet_indexes


    return points,indexes,F_in,F_out

def REV_division_TOTAL(meshQ,dir_results,Q_markers,radii,ind_bound,x,y,z,p_in,p_out):
    '''Function that divides a mesh growing in a cubic region in a specified number of equal REV, creating the corresponding submeshes and computing the pressure in each of them,testing the linear system method. Useful for mesh that are not cubic.
        INPUT:
            meshQ : mesh of the network saved as fenics mesh type.
            dir_results : directory in with the resulting mesh are solved.
            Q_markers : markers type that store mesh info.
            radii : radii associated with the edges of the network.
            ind_bound : indexes of the boundary nodes.
            x : delimiters of the sections in the x direction.
            y : delimiters of the sections in the y direction.
            z : delimiters of the sections in the z direction.
            p_in : Dirichlet condition for the inlet points.
            p_out : Dirichlet condition fot the outlet points.

        OUTPUT: 
            K_x : values of the x component of the upscaled permeability tensor (one for each REV).
            K_y : values of the y component of the upscaled permeability tensor (one for each REV).
            K_z : values of the z component of the upscaled permeability tensor (one for each REV).
            C : values of the RHS upscaled constant (one for each REV).
            mu_bl_up: values of the upscaled blood viscosity (one for each REV).
    '''
    coord=meshQ.coordinates()
    labels=Q_markers.array()
    edges=meshQ.cells()

    edges_old=np.copy(edges)
    righe_viste = set()
    edges = []

    for riga in edges_old:
        tupla_riga = tuple(riga)
        if tupla_riga not in righe_viste:
            righe_viste.add(tupla_riga)
            edges.append(riga)
    edges=np.array(edges)

    # print(np.shape(edges_old))
    # print(np.shape(edges))

    coord_old=np.copy(coord)

    righe_viste = set()
    coord = []

    for riga in coord_old:
        tupla_riga = tuple(riga)
        if tupla_riga not in righe_viste:
            righe_viste.add(tupla_riga)
            coord.append(riga)

    print(np.shape(coord_old))
    print(np.shape(coord))
    coord=np.array(coord)
    print('-------------------- MESH INFO --------------------')
    print(f'Coordinates: {np.shape(coord)}')
    print(f'Edges: {np.shape(edges)}')
    print(f'RAdii: {np.shape(radii)}')
    print(f'Labels: {np.shape(labels)}')
    print(f'Interior Points: {np.shape(np.where(labels==555)[0])}')
    print(f'Boundary Points: {np.shape(np.where(labels==111)[0])}')

    print()

    REV_division={}
    REV_labels=np.zeros(np.shape(labels))
    #np.shape(REV_labels)
    count=1
    sum=0
    tol2=1e-15
    
    xmin=np.min(coord[:,0])
    xmax=np.max(coord[:,0])
    ymin=np.min(coord[:,1])
    ymax=np.max(coord[:,1])
    zmin=np.min(coord[:,2])
    zmax=np.max(coord[:,2])
   
    x_planes=x[1:-1]
    y_planes=y[1:-1]
    z_planes=z[1:-1]

    for i in np.arange(len(x)-1):
        for j in np.arange(len(y)-1):
            for k in np.arange(len(z)-1):
                REV_division[count]={'xmin':np.maximum(xmin,x[i])}
                REV_division[count]['xmax']=np.minimum(xmax,x[i+1])
                REV_division[count]['ymin']=np.maximum(ymin,y[j])
                REV_division[count]['ymax']=np.minimum(ymax,y[j+1])
                REV_division[count]['zmin']=np.maximum(zmin,z[k])
                REV_division[count]['zmax']=np.minimum(zmax,z[k+1])

                indexes=np.where(((coord[:,0]-REV_division[count]['xmin'])>=-tol2) & ((coord[:,0]-REV_division[count]['xmax'])<=tol2) &
                    ((coord[:,1]-REV_division[count]['ymin'])>=-tol2) & ((coord[:,1]-REV_division[count]['ymax'])<=tol2) &
                ((coord[:,2]-REV_division[count]['zmin'])>=-tol2) & ((coord[:,2]-REV_division[count]['zmax'])<=tol2))[0]
            
                REV_division[count]['indexes']=indexes

                REV_labels[indexes]=count
                #
                # print(len(indexes))
                sum=sum+len(indexes)
                count=count+1
    print('-------------------- COMPUTING INTERSECTIONS --------------------')
    #FIND INTERSECTIONS
    coord_w_inter=np.copy(coord)
    edges_w_inter=np.copy(edges)
    radii_w_inter=np.copy(radii)
    REV_labels_w_inter=np.copy(REV_labels)


    res=np.where(REV_labels[edges[:,0]]!=REV_labels[edges[:,1]])[0]

    test_edge=edges[res]
    test_radii=radii[res]
    np.shape(test_edge)
    mask = np.ones(len(edges_w_inter), dtype=bool)
    mask[res] = False
    edges_w_inter = edges_w_inter[mask]
    radii_w_inter = radii_w_inter[mask]


    for rr,edge in enumerate(test_edge):
        dist_or=[]
        new_indexes=[]

        origin=coord[edge[0]]
        final=coord[edge[1]]
        res_tot=find_intersection2(coord[edge[0]],coord[edge[1]],x_planes,y_planes,z_planes)

        for i,res in enumerate(res_tot):
            dist_or.append(np.linalg.norm(origin-res))
            coord_w_inter=np.vstack([coord_w_inter,res])
            REV_labels_w_inter=np.append(REV_labels_w_inter,0)
            ind=np.where((coord_w_inter[:,0]==res[0]) & (coord_w_inter[:,1]==res[1]) & (coord_w_inter[:,2]==res[2]))[0][0]
            new_indexes.append(int(ind))

        dist_or.append(np.linalg.norm(origin-final))
        new_indexes.append(edge[1])

        #print(dist_or)
        res=np.argsort(dist_or)
        prev=edge[0]
        for r in res:
            if dist_or[r]!=0:
                edges_w_inter=np.vstack([edges_w_inter,[prev,new_indexes[r]]])
                radii_w_inter=np.append(radii_w_inter,test_radii[rr])
                prev=new_indexes[r]

    print(f'Coordinates with intersections: {np.shape(coord_w_inter)}')
    print(f'Edges with intersections: {np.shape(edges_w_inter)}')
    print(f'Labels with intersections: {np.shape(REV_labels_w_inter)}')
    print(f'Radii with intersections: {np.shape(radii_w_inter)}')

    
    print()
    print('... COMPUTING K VALUES  ...')
    print()

    init_REV=1
    nREV=count-1
    tol=1e-2/8
    tol2=1e-15
    sum=0
    

    name_mesh="CANCER"


    K_m1=np.zeros((count-1,))
    K_m2=np.zeros((count-1,))
    K_x=np.zeros((count-1,))
    K_y=np.zeros((count-1,))
    K_z=np.zeros((count-1,))

    C=np.zeros((count-1,))
    mu_bl_up_x=np.zeros((count-1,))
    mu_bl_up_y=np.zeros((count-1,))
    mu_bl_up_z=np.zeros((count-1,))

    for dir in [1,2,3]: 
        print(f'------------------------------------------------- DIRECTION: {dir}  -------------------------------------------------')
        for REV in np.arange(init_REV,nREV+1):
            print('-------------------------------------------------------------------------')

            print("REV = "+ str(REV))
            indexes=np.where(((coord_w_inter[:,0]-REV_division[REV]['xmin'])>=-tol2) & ((coord_w_inter[:,0]-REV_division[REV]['xmax'])<=tol2) &
                        ((coord_w_inter[:,1]-REV_division[REV]['ymin'])>=-tol2) & ((coord_w_inter[:,1]-REV_division[REV]['ymax'])<=tol2) &
                        ((coord_w_inter[:,2]-REV_division[REV]['zmin'])>=-tol2) & ((coord_w_inter[:,2]-REV_division[REV]['zmax'])<=tol2))[0]

            #EDGES
            edges_rev=[]
            radii_rev=[]

            index=np.where((REV_labels_w_inter[edges_w_inter[:,0]]==REV_labels_w_inter[edges_w_inter[:,1]]) & (REV_labels_w_inter[edges_w_inter[:,0]]==REV))[0]
            edges_rev_general=edges_w_inter[index]
            radii_rev_general=radii_w_inter[index]

            for rr,edge in enumerate(edges_rev_general):
                ind1=np.where(edge[0]==indexes)[0]
                ind2=np.where(edge[1]==indexes)[0]
                if len(ind1)!=0 and len(ind2)!=0:
                    edges_rev.append([ind1[0],ind2[0]])
                    radii_rev.append(radii_rev_general[rr])

            index=np.where((REV_labels_w_inter[edges_w_inter[:,0]]==0) & (REV_labels_w_inter[edges_w_inter[:,1]]==REV))[0]
            edges_rev_general1=edges_w_inter[index] 
            radii_rev_general1=radii_w_inter[index]

            for rr,edge in enumerate(edges_rev_general1):
                ind1=np.where(edge[0]==indexes)[0]
                ind2=np.where(edge[1]==indexes)[0]
                if len(ind1)!=0 and len(ind2)!=0:
                    edges_rev.append([ind1[0],ind2[0]])
                    radii_rev.append(radii_rev_general1[rr])

            index=np.where((REV_labels_w_inter[edges_w_inter[:,1]]==0) & (REV_labels_w_inter[edges_w_inter[:,0]]==REV))[0]
            edges_rev_general2=edges_w_inter[index]
            radii_rev_general2=radii_w_inter[index]
            

            for rr,edge in enumerate(edges_rev_general2):
                ind1=np.where(edge[0]==indexes)[0]
                ind2=np.where(edge[1]==indexes)[0]
                if len(ind1)!=0 and len(ind2)!=0:
                    edges_rev.append([ind1[0],ind2[0]])
                    radii_rev.append(radii_rev_general2[rr])

            index=np.where((REV_labels_w_inter[edges_w_inter[:,1]]==0) & (REV_labels_w_inter[edges_w_inter[:,0]]==0))[0]
            edges_rev_general3=edges_w_inter[index]
            radii_rev_general3=radii_w_inter[index]


            for rr,edge in enumerate(edges_rev_general3):
                ind1=np.where(edge[0]==indexes)[0]
                ind2=np.where(edge[1]==indexes)[0]

                if len(ind1)!=0 and len(ind2)!=0:
                    edges_rev.append([ind1[0],ind2[0]])
                    radii_rev.append(radii_rev_general3[rr])


            edges_rev=np.array(edges_rev)
            radii_rev=np.array(radii_rev)
            # print(np.shape(edges_rev))
            # print(np.shape(radii_rev))

            #VERTICES
            coord_rev=np.array(coord_w_inter[indexes])
            ind_bound_rev=[]
            for jj in ind_bound:
                check=np.where(jj==indexes)[0]
                if len(check)!=0:
                    ind_bound_rev.append(check)
            
            ind_bound_rev=np.array(ind_bound_rev)

            label_test=555*np.ones(np.shape(coord_rev)[0],)
            label_test[ind_bound_rev]=111

            #aux.plot_mesh(coord_rev,label_test,edges_rev,' Matrix Pressure REV '+str(REV))

            #LABELS
            labels_rev=555*np.ones((np.shape(coord_rev)[0],))

            scaling=1/1.05
            tol_edge_x=np.abs(np.mean(coord_rev[ind_bound_rev,0]-REV_division[REV]['xmin']))/scaling
            tol_edge_z=np.abs(np.mean(coord_rev[ind_bound_rev,2]-REV_division[REV]['zmin']))/scaling
            tol_edge_y=np.abs(np.mean(coord_rev[ind_bound_rev,1]-REV_division[REV]['ymin']))/scaling
            #print(tol_edge_x,tol_edge_y,tol_edge_z)

            if np.abs(xmin-REV_division[REV]['xmin'])<tol:
                #print('edge!')
                ind1_in=np.where(np.abs(coord_rev[ind_bound_rev,0]-REV_division[REV]['xmin'])<=tol_edge_x)[0]
                labels_rev[ind_bound_rev[ind1_in]]=111
            else:
                ind1_in=np.where(np.abs(coord_rev[:,0]-REV_division[REV]['xmin'])<tol)[0]
                labels_rev[ind1_in]=111


            if np.abs(ymin-REV_division[REV]['ymin'])<tol:
                #print('edge!')
                ind2_in=np.where(np.abs(coord_rev[ind_bound_rev,1]-REV_division[REV]['ymin'])<=tol_edge_y)[0]
                labels_rev[ind_bound_rev[ind2_in]]=112
            else:
                ind2_in=np.where(np.abs(coord_rev[:,1]-REV_division[REV]['ymin'])<tol)[0]
                labels_rev[ind2_in]=112


            if np.abs(zmin-REV_division[REV]['zmin'])<tol:
                #print('edge!')
                ind3_in=np.where(np.abs(coord_rev[ind_bound_rev,2]-REV_division[REV]['zmin'])<=tol_edge_z)[0]
                labels_rev[ind_bound_rev[ind3_in]]=113
            else:
                ind3_in=np.where(np.abs(coord_rev[:,2]-REV_division[REV]['zmin'])<tol)[0]
            
                labels_rev[ind3_in]=113

            #print(np.shape(ind1_in),np.shape(ind2_in),np.shape(ind3_in))
            tol_edge_x=np.abs(np.mean(coord_rev[ind_bound_rev,0]-REV_division[REV]['xmax']))
            tol_edge_z=np.abs(np.mean(coord_rev[ind_bound_rev,2]-REV_division[REV]['zmax']))
            tol_edge_y=np.abs(np.mean(coord_rev[ind_bound_rev,1]-REV_division[REV]['ymax']))
            #print(tol_edge_x,tol_edge_y,tol_edge_z)
            if np.abs(xmax-REV_division[REV]['xmax'])<tol:
                #print('edge!')
                ind1_out=np.where((np.abs(coord_rev[ind_bound_rev,0]-REV_division[REV]['xmax'])<=tol_edge_x))[0]
                labels_rev[ind_bound_rev[ind1_out]]=991
            else:
                ind1_out=np.where(np.abs(coord_rev[:,0]-REV_division[REV]['xmax'])<tol)[0]
                labels_rev[ind1_out]=991

            if np.abs(ymax-REV_division[REV]['ymax'])<tol:
                #print('edge!')
                ind2_out=np.where(np.abs(coord_rev[ind_bound_rev,1]-REV_division[REV]['ymax'])<=tol_edge_y)[0]
                labels_rev[ind_bound_rev[ind2_out]]=992
            else:
                ind2_out=np.where(np.abs(coord_rev[:,1]-REV_division[REV]['ymax'])<tol)[0]
                labels_rev[ind2_out]=992

            if np.abs(zmax-REV_division[REV]['zmax'])<tol:
                #print('edge!')
                ind3_out=np.where(np.abs(coord_rev[ind_bound_rev,2]-REV_division[REV]['zmax'])<=tol_edge_z)[0]
                labels_rev[ind_bound_rev[ind3_out]]=993

            else:
                ind3_out=np.where(np.abs(coord_rev[:,2]-REV_division[REV]['zmax'])<tol)[0]
                labels_rev[ind3_out]=993

            #aux.plot_mesh(coord_rev,labels_rev,edges_rev,' Matrix Pressure REV '+str(REV))

            create_mesh_REV(f"{dir_results}/TOTAL/TEST_LABEL"+ name_mesh +str(REV),coord_rev,edges_rev)


            meshQ2 = Mesh()
                
            with XDMFFile(f"{dir_results}/TOTAL/TEST_LABEL"+ name_mesh +str(REV)+".xdmf") as infile:
                infile.read(meshQ2)

            test_markers = MeshFunction('size_t', meshQ2, 0)

            test_markers.array()[:]=labels_rev

                    
            #PROCESSING EDGES
            ind_switch=np.where(((labels_rev[edges_rev[:,1]])==555) & ((labels_rev[edges_rev[:,0]])!=(labels_rev[edges_rev[:,1]])))[0]
            temp=np.copy(edges_rev[ind_switch,0])
            edges_rev[ind_switch,0]=edges_rev[ind_switch,1]
            edges_rev[ind_switch,1]=temp

            for ii in [111,112,113,991,992,993]:
                    ind=np.where(((labels_rev[edges_rev[:,1]])==ii) & ((labels_rev[edges_rev[:,0]])==(labels_rev[edges_rev[:,1]])))[0]
                    mask = np.ones(len(edges_rev), dtype=bool)
                    mask[ind] = False
                    edges_rev = edges_rev[mask]
                    radii_rev = radii_rev[mask]


            G,connected_components=compute_connected_components(edges_rev)

            comp1 = max(connected_components, key=len)

            nodes_comp1=np.array(G.subgraph(comp1).nodes())
            temp_edges=np.array(G.subgraph(comp1).edges())
            #print(edges_rev)

            #ADATTARE GLI EDGES AI NUOVI NODI
            coord_comp1=coord_rev[nodes_comp1]
            labels_comp1=labels_rev[nodes_comp1]
            edges_comp1=[]
            radii_comp1=[]
            for edge in temp_edges:
                    ind1=np.where((coord_comp1[:,0]==coord_rev[edge[0],0]) &(coord_comp1[:,1]==coord_rev[edge[0],1]) & (coord_comp1[:,2]==coord_rev[edge[0],2]) )[0]
                    ind2=np.where((coord_comp1[:,0]==coord_rev[edge[1],0]) &(coord_comp1[:,1]==coord_rev[edge[1],1]) & (coord_comp1[:,2]==coord_rev[edge[1],2]) )[0]
                        
                    edges_comp1.append([ind1[0],ind2[0]])
                    ind=np.where((edges_rev[:,0]==edge[0]) & (edges_rev[:,1]==edge[1]))[0]
                    if len(ind)!=0:
                            radii_comp1.append(radii_rev[ind][0])
                    else:
                            np.random.seed(42)
                            min_val=np.min(radii_w_inter)
                            max_val=np.max(radii_w_inter)
                            value = np.random.uniform(min_val, max_val)
                            radii_comp1.append(value)
                    # print(np.shape(radii_comp1))
                    # print(np.shape(edges_comp1))
                    #print(radii_comp1)

            edges_comp1=np.array(edges_comp1)
            radii_comp1=np.array(radii_comp1)

            #PROCESSING EDGES
            ind_switch=np.where(((labels_comp1[edges_comp1[:,1]])==555) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            for ii in [111,112,113,991,992,993]:
                ind=np.where(((labels_comp1[edges_comp1[:,1]])==ii) & ((labels_comp1[edges_comp1[:,0]])==(labels_comp1[edges_comp1[:,1]])))[0]
                mask = np.ones(len(edges_comp1), dtype=bool)
                mask[ind] = False
                edges_comp1 = edges_comp1[mask]
                radii_comp1 = radii_comp1[mask]

            #PROCESSING EDGES
            ind_switch=np.where(((labels_comp1[edges_comp1[:,0]])==112) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            #PROCESSING EDGES

            if dir==1:
                indx_in=111
                indx_out=991

            if dir==2:
                indx_in=112
                indx_out=992

            if dir==3:
                indx_in=113
                indx_out=993

            ind_switch=np.where(((labels_comp1[edges_comp1[:,0]])==indx_in) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            ind_switch=np.where(((labels_comp1[edges_comp1[:,0]])==indx_out) & ((labels_comp1[edges_comp1[:,0]])!=(labels_comp1[edges_comp1[:,1]])))[0]
            temp=np.copy(edges_comp1[ind_switch,0])
            edges_comp1[ind_switch,0]=edges_comp1[ind_switch,1]
            edges_comp1[ind_switch,1]=temp

            ind_change=np.where(((labels_comp1[edges_comp1[:,1]])==indx_in) & ((labels_comp1[edges_comp1[:,0]]!=555)))[0]
            for temp_ed in edges_comp1[ind_change]:
                i=temp_ed[0]
                ind=np.where((edges_comp1[:,0]==i) | (edges_comp1[:,1]==i))[0]
                if len(ind)!=1:
                    labels_comp1[i]=555

            ind_change=np.where(((labels_comp1[edges_comp1[:,1]])==indx_out) & ((labels_comp1[edges_comp1[:,0]]!=555)))[0]
            for temp_ed in edges_comp1[ind_change]:
                i=temp_ed[0]
                ind=np.where((edges_comp1[:,0]==i) | (edges_comp1[:,1]==i))[0]
                if len(ind)!=1:
                    labels_comp1[i]=555


            labels_comp1_total=np.copy(labels_comp1)

            ind_out_total=np.where((labels_comp1==991) | (labels_comp1==992) | (labels_comp1==993))[0]
            labels_comp1_total[ind_out_total]=999


            ind_in_total=np.where((labels_comp1==111) | (labels_comp1==112) | (labels_comp1==113))[0]
            labels_comp1_total[ind_in_total]=111

            print('-------------------- SUBMESH INFO --------------------')
            print(f'Coordinates: {np.shape(coord_comp1)}')
            print(f'Edges: {np.shape(edges_comp1)}')
            print(f'Labels: {np.shape(labels_comp1_total)}')
            print(f'Inlet points submesh: {len(ind_in_total)}')
            print(f'Outlet points submesh: {len(ind_out_total)}')

        
            

            create_mesh_REV(f"{dir_results}/TOTAL/comp_conn_REV_"+ name_mesh +str(REV),coord_comp1,edges_comp1)


            points_comp1,indexes_comp1,Fin_comp1,Fout_comp1=compute_faces_TOTAL(labels_comp1_total,labels_comp1,coord_comp1)
            

            meshQ2 = Mesh()
                    
            with XDMFFile(f"{dir_results}/TOTAL/comp_conn_REV_"+ name_mesh +str(REV)+".xdmf") as infile:
                infile.read(meshQ2)

            if dir==1:
                dir_str='x'

            if dir==2:
                dir_str='y'

            if dir==3:
                dir_str='z'

            #print(dir,dir_str)

            

            p_tot=compute_pressure2(dir,meshQ2,edges_comp1,coord_comp1,labels_comp1_total, radii_comp1, indexes_comp1['inlet']['total'],indexes_comp1['inlet'][dir_str],indexes_comp1['outlet']['total'],indexes_comp1['outlet'][dir_str],Fin_comp1,Fout_comp1,p_in,p_out)



            p_tot_lap=compute_pressure(meshQ2,points_comp1['inlet'][dir_str],points_comp1['outlet'][dir_str],p_in,p_out)
            sol_val=p_tot_lap.compute_vertex_values(meshQ2)

            # diff_p=sol_val-p_tot
            # ind=np.where(diff_p>1e-10)[0]
            # print(diff_p[ind])
            # print(p_tot[ind])
            # print(sol_val[ind])

            k_tot, mu_bl_up=compute_k(dir,meshQ2,radii_comp1,indexes_comp1['outlet'][dir_str],p_tot,coord_comp1,edges_comp1,p_in,p_out)

            k_sol_val, mu_bl_up=compute_k(dir,meshQ2,radii_comp1,indexes_comp1['outlet'][dir_str],sol_val,coord_comp1,edges_comp1,p_in,p_out)

            C_j=compute_C_value(meshQ2,edges_comp1,coord_comp1,radii_comp1,REV_division[REV]['xmax'],REV_division[REV]['xmin'],REV_division[REV]['ymax'],REV_division[REV]['ymin'],REV_division[REV]['zmax'],REV_division[REV]['zmin'])

            K_m1[REV-1]=np.abs(k_tot)
            K_m2[REV-1]=np.abs(k_sol_val)

            if dir==1:
                K_x[REV-1]=np.abs(k_tot)
                mu_bl_up_x[REV-1]=np.abs(mu_bl_up)

            if dir==2:
                K_y[REV-1]=np.abs(k_tot)
                mu_bl_up_y[REV-1]=np.abs(mu_bl_up)

            if dir==3:
                K_z[REV-1]=np.abs(k_tot)
                mu_bl_up_z[REV-1]=np.abs(mu_bl_up)


            C[REV-1]=C_j

            print(k_tot)
            print(k_sol_val)
            print(C_j)
            mu_bl_up=mu_bl_up_x

    return K_x,K_y,K_z,C,mu_bl_up
            

def generate_radom_raddii(meshQ,min_val,max_val):
    #Generates random radii between a minimum and a maximum value to associate to each edge.
    np.random.seed(42)
    vector = []
    n=len(meshQ.cells())
    for _ in range(n):
        value = np.random.uniform(min_val, max_val)
        vector.append(value)
    radii = np.array(vector)
    return radii

def create_dictionary(meshV,n_div_x,n_div_y,n_div_z):
    #Creates the dictionary that stores the points in each REV for cubic mesh
    REV_division={}
    coord=meshV.coordinates()

    REV_labels=np.zeros(np.shape(coord)[0],)
    count=1
    tol2=1e-15
    xmin=np.min(coord[:,0])
    xmax=np.max(coord[:,0])
    ymin=np.min(coord[:,1])
    ymax=np.max(coord[:,1])
    zmin=np.min(coord[:,2])
    zmax=np.max(coord[:,2])

    x= np.linspace(xmin,xmax,n_div_x+1)
    y= np.linspace(ymin,ymax,n_div_y+1)
    z=np.linspace(zmin,zmax,n_div_z+1)

    for i in np.arange(len(x)-1):
        for j in np.arange(len(y)-1):
            for k in np.arange(len(z)-1):
                REV_division[count]={'xmin':np.maximum(xmin,x[i])}
                REV_division[count]['xmax']=np.minimum(xmax,x[i+1])
                REV_division[count]['ymin']=np.maximum(ymin,y[j])
                REV_division[count]['ymax']=np.minimum(ymax,y[j+1])
                REV_division[count]['zmin']=np.maximum(zmin,z[k])
                REV_division[count]['zmax']=np.minimum(zmax,z[k+1])

                indexes=np.where(((coord[:,0]-REV_division[count]['xmin'])>=-tol2) & ((coord[:,0]-REV_division[count]['xmax'])<=tol2) &
                    ((coord[:,1]-REV_division[count]['ymin'])>=-tol2) & ((coord[:,1]-REV_division[count]['ymax'])<=tol2) &
                    ((coord[:,2]-REV_division[count]['zmin'])>=-tol2) & ((coord[:,2]-REV_division[count]['zmax'])<=tol2))[0]
                
                REV_division[count]['indexes']=indexes

                REV_labels[indexes]=count
                count=count+1
                
    return REV_division

def create_dictionary_TOTAL(meshV,x,y,z):
    #Creates the dictionary that stores the points in each REV for general mesh

    REV_division={}
    coord=meshV.coordinates()

    REV_labels=np.zeros(np.shape(coord)[0],)
    count=1
    tol2=1e-15
    xmin=np.min(coord[:,0])
    xmax=np.max(coord[:,0])
    ymin=np.min(coord[:,1])
    ymax=np.max(coord[:,1])
    zmin=np.min(coord[:,2])
    zmax=np.max(coord[:,2])

    # x= np.linspace(xmin,xmax,n_div_x+1)
    # y= np.linspace(ymin,ymax,n_div_y+1)
    # z=np.linspace(zmin,zmax,n_div_z+1)

    for i in np.arange(len(x)-1):
        for j in np.arange(len(y)-1):
            for k in np.arange(len(z)-1):
                REV_division[count]={'xmin':np.maximum(xmin,x[i])}
                REV_division[count]['xmax']=np.minimum(xmax,x[i+1])
                REV_division[count]['ymin']=np.maximum(ymin,y[j])
                REV_division[count]['ymax']=np.minimum(ymax,y[j+1])
                REV_division[count]['zmin']=np.maximum(zmin,z[k])
                REV_division[count]['zmax']=np.minimum(zmax,z[k+1])

                indexes=np.where(((coord[:,0]-REV_division[count]['xmin'])>=-tol2) & ((coord[:,0]-REV_division[count]['xmax'])<=tol2) &
                    ((coord[:,1]-REV_division[count]['ymin'])>=-tol2) & ((coord[:,1]-REV_division[count]['ymax'])<=tol2) &
                    ((coord[:,2]-REV_division[count]['zmin'])>=-tol2) & ((coord[:,2]-REV_division[count]['zmax'])<=tol2))[0]
                
                REV_division[count]['indexes']=indexes

                REV_labels[indexes]=count
                count=count+1
                
    return REV_division
            
            


        










            


