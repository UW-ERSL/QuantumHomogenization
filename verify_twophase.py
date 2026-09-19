import numpy as np
from qhomogenize.task2.build_UDp import build_UMp
from qhomogenize.task2.build_UDk import build_UDk
from qhomogenize.task1.build_UA import build_UA
from qhomogenize.task2.square_indicator import square_indicator_array
from qhomogenize.fea.element_template import element_template
from qhomogenize.fea.assembly_pbc import assembly_matrix

def lame(E,nu): return E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))

for L in [2,4]:
    sx,sy,side = (0,0,1) if L==2 else (1,1,2)
    E0,nu0,E1,nu1=1.0,0.3,0.1,0.3
    lam0,mu0=lame(E0,nu0); lam1,mu1=lame(E1,nu1)
    a_h=1/(2*L)
    Kl,Km,_,_=element_template(a=a_h,b=a_h,phi_deg=90.0)

    # p in pilot order e=ix+L*iy
    p_arr=square_indicator_array(L,sx,sy,side)
    p_vec=np.zeros(L*L)
    for ix in range(L):
        for iy in range(L):
            p_vec[ix+L*iy]=p_arr[iy,ix]

    # Extract small primitive blocks (all <= 11 qubits => fine)
    be_Mp=build_UMp(L,"square",dict(sx=sx,sy=sy,side_x=side))
    be_Dlam=build_UDk(L,a=a_h,b=a_h,lam=1.0,mu=0.0)
    be_Dmu =build_UDk(L,a=a_h,b=a_h,lam=0.0,mu=1.0)
    be_A=build_UA(L); be_AT=be_A.conjugate_transpose()

    Mp = be_Mp.extract_matrix()*be_Mp.alpha          # raw M_p
    Dl = be_Dlam.extract_matrix()*be_Dlam.alpha      # raw D_lam
    Dm = be_Dmu.extract_matrix()*be_Dmu.alpha        # raw D_mu
    Araw = be_A.extract_matrix()*be_A.alpha
    ATraw= be_AT.extract_matrix()*be_AT.alpha

    # Recompose Dp classically from the quantum-extracted primitives
    Dp_q = lam0*Dl + mu0*Dm + (lam1-lam0)*(Mp@Dl) + (mu1-mu0)*(Mp@Dm)
    K_q  = ATraw @ Dp_q @ Araw                       # full padded 8L^2 sq

    # Classical reference (independent assembly)
    Dl_ref=np.kron(np.eye(L*L),Kl); Dm_ref=np.kron(np.eye(L*L),Km)
    Mp_ref=np.kron(np.diag(p_vec),np.eye(8))
    Dp_ref=lam0*Dl_ref+mu0*Dm_ref+(lam1-lam0)*Mp_ref@Dl_ref+(mu1-mu0)*Mp_ref@Dm_ref
    A=assembly_matrix(L).toarray(); nloc,nglob=A.shape
    Apad=np.zeros((nloc,nloc)); Apad[:,:nglob]=A
    K_ref=Apad.T@Dp_ref@Apad

    na=2*L*L
    eMp=np.max(np.abs(Mp-Mp_ref)); eDp=np.max(np.abs(Dp_q-Dp_ref))
    eK =np.max(np.abs(K_q[:na,:na]-K_ref[:na,:na]))
    print(f"L={L} square(sx={sx},sy={sy},side={side}):")
    print(f"   max|Mp_q - Mp_ref|  = {eMp:.2e}")
    print(f"   max|Dp_q - Dp_ref|  = {eDp:.2e}")
    print(f"   max|K_q  - K_ref|   = {eK:.2e}   (active {na}x{na} block)")
    # alpha factorization for the assembled LCU encoding
    alpha_Dp = abs(lam0)*be_Dlam.alpha+abs(mu0)*be_Dmu.alpha+abs(lam1-lam0)*1.0*be_Dlam.alpha+abs(mu1-mu0)*1.0*be_Dmu.alpha
    print(f"   alpha_Dp(LCU)={alpha_Dp:.4f}  alpha_K=alpha_A^2*alpha_Dp={be_A.alpha**2*alpha_Dp:.4f}")
