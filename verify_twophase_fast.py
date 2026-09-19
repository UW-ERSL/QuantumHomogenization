import numpy as np, time
from qhomogenize.bencode.extract_fast import extract_block
from qhomogenize.task2.build_UDp import build_UMp
from qhomogenize.task2.build_UDk import build_UDk, build_UKe
from qhomogenize.task1.build_UA import build_UA
from qhomogenize.task2.square_indicator import square_indicator_array
from qhomogenize.fea.element_template import element_template
from qhomogenize.fea.assembly_pbc import assembly_matrix

def lame(E,nu): return E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))

def verify(L, sx, sy, side):
    E0,nu0,E1,nu1=1.0,0.3,0.1,0.3
    lam0,mu0=lame(E0,nu0); lam1,mu1=lame(E1,nu1)
    a_h=1/(2*L)
    Kl,Km,_,_=element_template(a=a_h,b=a_h,phi_deg=90.0)

    p_arr=square_indicator_array(L,sx,sy,side)
    p_vec=np.zeros(L*L)
    for ix in range(L):
        for iy in range(L): p_vec[ix+L*iy]=p_arr[iy,ix]

    # U_A: needs simulation -> fast extractor (many ancillas, big win)
    be_A=build_UA(L); be_AT=be_A.conjugate_transpose()
    Araw  = extract_block(be_A).real * be_A.alpha
    ATraw = Araw.T.copy()   # A^T block = transpose of A block (real)

    # D_lam, D_mu: analytic tensor I_{L^2} (x) K_e^layer (no big simulation)
    be_Ke_l = build_UKe(a=a_h,b=a_h,lam=1.0,mu=0.0)
    be_Ke_m = build_UKe(a=a_h,b=a_h,lam=0.0,mu=1.0)
    Ke_l_blk = extract_block(be_Ke_l).real * be_Ke_l.alpha   # 8x8
    Ke_m_blk = extract_block(be_Ke_m).real * be_Ke_m.alpha
    Dl = np.kron(np.eye(L*L), Ke_l_blk)
    Dm = np.kron(np.eye(L*L), Ke_m_blk)

    # M_p: diagonal projector -> small simulation (8 sys + few anc) or analytic
    be_Mp=build_UMp(L,"square",dict(sx=sx,sy=sy,side_x=side))
    Mp = extract_block(be_Mp).real * be_Mp.alpha

    Dp_q = lam0*Dl + mu0*Dm + (lam1-lam0)*(Mp@Dl) + (mu1-mu0)*(Mp@Dm)
    K_q  = ATraw @ Dp_q @ Araw

    # classical ref
    Mp_ref=np.kron(np.diag(p_vec),np.eye(8))
    Dl_ref=np.kron(np.eye(L*L),Kl); Dm_ref=np.kron(np.eye(L*L),Km)
    Dp_ref=lam0*Dl_ref+mu0*Dm_ref+(lam1-lam0)*Mp_ref@Dl_ref+(mu1-mu0)*Mp_ref@Dm_ref
    A=assembly_matrix(L).toarray(); nloc,nglob=A.shape
    Apad=np.zeros((nloc,nloc)); Apad[:,:nglob]=A
    K_ref=Apad.T@Dp_ref@Apad
    na=2*L*L
    return (np.max(np.abs(Mp-Mp_ref)), np.max(np.abs(Dp_q-Dp_ref)),
            np.max(np.abs(K_q[:na,:na]-K_ref[:na,:na])), be_A.alpha**2*(
            abs(lam0)*be_Ke_l.alpha+abs(mu0)*be_Ke_m.alpha+abs(lam1-lam0)*be_Ke_l.alpha+abs(mu1-mu0)*be_Ke_m.alpha))

for L,(sx,sy,side) in [(4,(1,1,2)),(8,(2,2,4))]:
    t0=time.time()
    eMp,eDp,eK,aK=verify(L,sx,sy,side)
    print(f"L={L}: max|Mp|={eMp:.1e} max|Dp|={eDp:.1e} max|K|={eK:.1e} alpha_K={aK:.4f}  [{time.time()-t0:.1f}s]")
