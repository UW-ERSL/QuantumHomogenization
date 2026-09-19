import numpy as np, sys, time
sys.path.insert(0,"/mnt/project")
from qhomogenize.bencode.extract_fast import extract_block
from qhomogenize.task2.build_UDk import build_UKe
from qhomogenize.task1.build_UA import build_UA
from qhomogenize.task2.build_UDp import build_UMp
from qhomogenize.task2.square_indicator import square_indicator_array
from Chapter03_EngineeringOptimization_functions import FEA2DHomogenize as CH, MicrostructureGenerator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def homogenize_from_quantum(L, micro, E0, nu0, E1, nu1):
    """Replicate chapter fea_homogenize, but build K and F from QUANTUM-extracted
    element template (U_Ke) and quantum assembly structure. micro[iy,ix] in {0,1},
    convention: micro==0 -> matrix (E0), micro==1 -> inclusion (E1) -- matching chapter line 2048."""
    nelx=nely=L; nel=L*L; lx=ly=L
    dx=lx/nelx; dy=ly/nely
    # --- QUANTUM element template at chapter scaling a=dx/2 ---
    a_h=dx/2; b_h=dy/2
    be_Kl=build_UKe(a=a_h,b=b_h,lam=1.0,mu=0.0)
    be_Km=build_UKe(a=a_h,b=b_h,lam=0.0,mu=1.0)
    ke_lambda=(extract_block(be_Kl).real*be_Kl.alpha)
    ke_mu    =(extract_block(be_Km).real*be_Km.alpha)
    # load vectors fe come from the SAME element routine (not separately encoded);
    # take them from the classical element_mat_vec at the same scaling (they are
    # part of the element definition, not the operator being encoded).
    _,_,fe_lambda,fe_mu = CH.element_mat_vec(a_h,b_h,90.0)

    # --- chapter DOF / PBC bookkeeping (verbatim) ---
    nodenrs=np.arange(1,(1+nelx)*(1+nely)+1).reshape(1+nely,1+nelx,order='F')
    edof_vec=(2*nodenrs[:-1,:-1]+1).flatten(order='F')
    edof_mat=np.tile(edof_vec.reshape(-1,1),(1,8))+np.tile(np.array([0,1,2*nely+2,2*nely+3,2*nely,2*nely+1,-2,-1]),(nel,1))
    nn=(nelx+1)*(nely+1); nnP=nelx*nely
    arr=np.arange(1,nnP+1).reshape(nely,nelx,order='F'); arr=np.vstack([arr,arr[0,:]]); arr=np.column_stack([arr,arr[:,0]])
    dofv=np.zeros(2*nn,dtype=int); dofv[0::2]=2*arr.flatten(order='F')-1; dofv[1::2]=2*arr.flatten(order='F')
    edof_mat=dofv[edof_mat-1]; ndof=2*nnP
    iK=np.kron(edof_mat,np.ones((8,1))).T; jK=np.kron(edof_mat,np.ones((1,8))).T
    E=np.array([E0,E1]); nu=np.array([nu0,nu1])
    lam=E*nu/((1+nu)*(1-2*nu)); mu=E/(2*(1+nu))
    lam_el=lam[0]*(micro==0)+lam[1]*(micro==1)
    mu_el =mu[0]*(micro==0)+mu[1]*(micro==1)
    sK=np.outer(ke_lambda.flatten(order='F'),lam_el.flatten(order='F'))+np.outer(ke_mu.flatten(order='F'),mu_el.flatten(order='F'))
    K=coo_matrix((sK.flatten(order='F'),(iK.flatten(order='F')-1,jK.flatten(order='F')-1)),shape=(ndof,ndof)).tocsr()
    sF=np.outer(fe_lambda.flatten(order='F'),lam_el.flatten(order='F'))+np.outer(fe_mu.flatten(order='F'),mu_el.flatten(order='F'))
    iF=np.tile(edof_mat.T,(3,1)); jF=np.vstack([np.ones((8,nel)),2*np.ones((8,nel)),3*np.ones((8,nel))])
    F=coo_matrix((sF.flatten(order='F'),(iF.flatten(order='F')-1,jF.flatten(order='F')-1)),shape=(ndof,3)).toarray()
    chi=np.zeros((ndof,3)); chi[2:,:]=spsolve(K[2:,2:].tocsr(),F[2:,:])
    chi0=np.zeros((nel,8,3)); chi0_e=np.zeros((8,3))
    ke=ke_mu+ke_lambda; fe=fe_mu+fe_lambda
    fd=np.array([2,4,5,6,7]); chi0_e[fd,:]=np.linalg.solve(ke[np.ix_(fd,fd)],fe[fd,:])
    for i in range(3): chi0[:,:,i]=np.tile(chi0_e[:,i],(nel,1))
    CHt=np.zeros((3,3)); vol=lx*ly
    for i in range(3):
        for j in range(3):
            ci=chi0[:,:,i]-chi[edof_mat-1,i]; cj=chi0[:,:,j]-chi[edof_mat-1,j]
            sl=np.sum((ci@ke_lambda)*cj,axis=1).reshape(nely,nelx,order='F')
            sm=np.sum((ci@ke_mu)*cj,axis=1).reshape(nely,nelx,order='F')
            CHt[i,j]=(1/vol)*np.sum(lam_el*sl+mu_el*sm)
    return CHt

for L in [4,8]:
    micro=MicrostructureGenerator(L,L,inclusion_fraction=0.25,micro_type='square').data
    t0=time.time()
    CH_q  = homogenize_from_quantum(L,micro,1.0,0.3,0.1,0.3)
    CH_ref= CH.fea_homogenize(E_incl=0.1,nu_incl=0.3,E_matrix=1.0,nu_matrix=0.3,microData=micro)
    err=np.max(np.abs(CH_q-CH_ref))
    print(f"L={L}: max|C^H_quantum - C^H_chapter| = {err:.3e}   [{time.time()-t0:.1f}s]")
    print("  C^H_quantum =\n", np.round(CH_q,5))
