import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from matplotlib.path import Path as Pathm
from scipy.interpolate import griddata
from typing import Dict, Tuple
import scipy.linalg as la
import scipy.sparse as sp


def shape_functions(xi: float, eta: float):
    """Return shape functions **N** (length 4) and derivatives dN/d{xi,eta}."""
    N = 0.25 * np.array(
        [
            (1 - xi) * (1 + eta),
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
        ]
    )
    dN_dxi = 0.25 * np.array(
        [
            [-(1 + eta), +(1 - xi)],
            [-(1 - eta), -(1 - xi)],
            [+(1 - eta), -(1 + xi)],
            [+(1 + eta), +(1 + xi)],
        ]
    )
    return N, dN_dxi
 
 
def strain_matrix(dNdx: np.ndarray, nlgeom: bool = False):
    """Assemble the 3×8 (or 4×8, large‑strain) strain‑displacement matrix."""
    B = np.array(
        [
            [
                dNdx[0, 0],
                0,
                dNdx[1, 0],
                0,
                dNdx[2, 0],
                0,
                dNdx[3, 0],
                0,
            ],
            [
                0,
                dNdx[0, 1],
                0,
                dNdx[1, 1],
                0,
                dNdx[2, 1],
                0,
                dNdx[3, 1],
            ],
            [
                dNdx[0, 1],
                dNdx[0, 0],
                dNdx[1, 1],
                dNdx[1, 0],
                dNdx[2, 1],
                dNdx[2, 0],
                dNdx[3, 1],
                dNdx[3, 0],
            ],
        ]
    )
    if nlgeom:
        B = np.vstack(
            [
                B,
                [
                    0,
                    dNdx[0, 0],
                    0,
                    dNdx[1, 0],
                    0,
                    dNdx[2, 0],
                    0,
                    dNdx[3, 0],
                ],
            ]
        )
    return B
 
 
# ------------------------------------------------------------------
# inverse transform (Hua 1990 + Newton fallback)
# ------------------------------------------------------------------
 
def inverse_transform(node_xy: np.ndarray, pt_xy: np.ndarray, tol: float = 1e-12):
    """Global ➜ local (xi,eta) mapping inside a 4‑noded quad (Hua, 1990)."""
    x, y = pt_xy
    A = np.array([[-1, 1, -1, 1], [-1, -1, 1, 1], [1, -1, -1, 1]]) @ node_xy
    d = 4 * np.array([x, y]) - node_xy.sum(axis=0)
 
    a1, a2 = A[0]
    b1, b2 = A[1]
    c1, c2 = A[2]
    d1, d2 = d
 
    ab = a1 * b2 - a2 * b1
    ac = a1 * c2 - a2 * c1
    cb = c1 * b2 - c2 * b1
    da = d1 * a2 - d2 * a1
    dc = d1 * c2 - d2 * c1
 
    # generic (non‑degenerate) case – direct quadratic roots
    if abs(ab) > tol and abs(ac) > tol:
        roots = np.roots([ab, cb + da, dc])
        roots = roots[np.isreal(roots)].real
        inside = roots[np.abs(roots) <= 1 + tol]
        if inside.size:
            xi = inside[0]
            eta = (da + (b1 * a2 - b2 * a1) * xi) / ac
            return xi, eta
 
    # Newton fallback (robust for distorted quads)
    xi = eta = 0.0
    for _ in range(8):
        N, dN_dxi = shape_functions(xi, eta)
        xy = (N[:, None] * node_xy).sum(axis=0)
        res = np.array([x - xy[0], y - xy[1]])
        if np.linalg.norm(res) < tol:
            break
        J = dN_dxi.T @ node_xy
        dxi_deta = la.solve(J, res)
        xi += dxi_deta[0]
        eta += dxi_deta[1]
    return xi, eta
 
def compute_mesh_grids(X: np.ndarray, Y: np.ndarray):
    """
    One–to‑one port of MATLAB *computeMeshGrids*.
    The only difference w.r.t. the version you posted is the explicit,
    deterministic extrapolation of the outermost ring of nodes so that they
    sit exactly half a pitch outside the last interior row/column.
    """
    ny, nx = X.shape
    MeshX = np.full((ny + 1, nx + 1), np.nan, dtype=float)
    MeshY = np.full_like(MeshX, np.nan)
 
    # --- step 1 – interior nodes (identical to MATLAB) --------------------
    avg_dx = np.nanmean(np.diff(X, axis=1))
    avg_dy = np.nanmean(np.diff(Y, axis=0))
 
    for i in range(0, ny):
        for j in range(0, nx):
            dx = X[i, j] - X[i, j - 1]
            dy = Y[i, j] - Y[i - 1, j]
            if np.isnan(dx):
                dx = avg_dx
            if np.isnan(dy):
                dy = avg_dy
 
            x_coord = np.nanmean([X[i, j] - dx * 0.5,
                                  X[i, j] + dx * 0.5])
            y_coord = np.nanmean([Y[i, j] - dy * 0.5,
                                  Y[i, j] + dy * 0.5])
 
            MeshX[i, j] = MeshX[i, j] = x_coord
            MeshY[i, j] = MeshY[i, j] = y_coord
 
    # --- step 2 – *linear* extrapolation of the outer ring ---------------
    # rows ---------------------------------------------------------------
    MeshX[0, 1:-1]   = 2*MeshX[1, 1:-1]   - MeshX[2, 1:-1]   # top
    MeshX[-1, 1:-1]  = 2*MeshX[-2, 1:-1]  - MeshX[-3, 1:-1]  # bottom
    MeshY[0, 1:-1]   = 2*MeshY[1, 1:-1]   - MeshY[2, 1:-1]
    MeshY[-1, 1:-1]  = 2*MeshY[-2, 1:-1]  - MeshY[-3, 1:-1]
 
    # columns ------------------------------------------------------------
    MeshX[1:-1, 0]   = 2*MeshX[1:-1, 1]   - MeshX[1:-1, 2]   # left
    MeshX[1:-1, -1]  = 2*MeshX[1:-1, -2]  - MeshX[1:-1, -3]  # right
    MeshY[1:-1, 0]   = 2*MeshY[1:-1, 1]   - MeshY[1:-1, 2]
    MeshY[1:-1, -1]  = 2*MeshY[1:-1, -2]  - MeshY[1:-1, -3]
 
    # corners ------------------------------------------------------------
    MeshX[0, 0]      = MeshX[0, 1]  + MeshX[1, 0]  - MeshX[1, 1]
    MeshX[0, -1]     = MeshX[0, -2] + MeshX[1, -1] - MeshX[1, -2]
    MeshX[-1, 0]     = MeshX[-1, 1] + MeshX[-2, 0] - MeshX[-2, 1]
    MeshX[-1, -1]    = MeshX[-1, -2]+ MeshX[-2, -1]- MeshX[-2, -2]
 
    MeshY[0, 0]      = MeshY[0, 1]  + MeshY[1, 0]  - MeshY[1, 1]
    MeshY[0, -1]     = MeshY[0, -2] + MeshY[1, -1] - MeshY[1, -2]
    MeshY[-1, 0]     = MeshY[-1, 1] + MeshY[-2, 0] - MeshY[-2, 1]
    MeshY[-1, -1]    = MeshY[-1, -2]+ MeshY[-2, -1]- MeshY[-2, -2]
 
    # --- step 3 – fill any corner nodes that are still nan -------------------
    missing = np.isnan(MeshX) | np.isnan(MeshY)
    if np.any(missing):
        yi, xi = np.where(~missing)               # coordinates of known nodes
        pts = np.column_stack((xi, yi))
        # --- X ---------------------------------------------------------------
        MeshX[missing] = griddata(
            pts, MeshX[~missing], (np.where(missing)[1], np.where(missing)[0]),
            method='linear', fill_value=np.nan, rescale=True)
        # still nan? – nearest fallback
        bad = np.isnan(MeshX)
        if np.any(bad):
            MeshX[bad] = griddata(
                pts, MeshX[~missing], (np.where(bad)[1], np.where(bad)[0]),
                method='nearest', rescale=True)
        # --- Y ---------------------------------------------------------------
        MeshY[missing] = griddata(
            pts, MeshY[~missing], (np.where(missing)[1], np.where(missing)[0]),
            method='linear', fill_value=np.nan, rescale=True)
        bad = np.isnan(MeshY)
        if np.any(bad):
            MeshY[bad] = griddata(
                pts, MeshY[~missing], (np.where(bad)[1], np.where(bad)[0]),
                method='nearest', rescale=True)
 
    return MeshX, MeshY
 
# ------------------------------------------------------------------
# seedVirtualNodes – exact MATLAB logic, incl. snapping to centroids
# ------------------------------------------------------------------
 
def seed_virtual_nodes(
    nElem: tuple[int, int],
    meshX: np.ndarray,
    meshY: np.ndarray,
    y_down_decrease_flag: bool,
):
    """Return the 1‑D arrays Xvec, Yvec of virtual‑node coordinates."""
 
    pt_pos_x = np.sort(np.unique(meshX[~np.isnan(meshX)]))
 
    if y_down_decrease_flag:  # Cartesian (Y decreases down the image)
        pt_pos_y = np.sort(np.unique(meshY[~np.isnan(meshY)]))[::-1]
    else:  # image convention (Y increases downward)
        pt_pos_y = np.sort(np.unique(meshY[~np.isnan(meshY)]))
 
    # initial equally‑spaced guess
    Xvec = np.linspace(pt_pos_x[0], pt_pos_x[-1], nElem[0] + 1)
    Yvec = np.linspace(pt_pos_y[0], pt_pos_y[-1], nElem[1] + 1)
 
    # snap interior nodes to the nearest centroid coordinate (like MATLAB loop)
    for i in range(1, len(Xvec) - 1):
        idx = np.nanargmin(np.abs(pt_pos_x[1:-1] - Xvec[i]))
        Xvec[i] = pt_pos_x[1:-1][idx]
 
    for i in range(1, len(Yvec) - 1):
        idx = np.nanargmin(np.abs(pt_pos_y[1:-1] - Yvec[i]))
        Yvec[i] = pt_pos_y[1:-1][idx]
 
    return Xvec, Yvec
 
 
# ------------------------------------------------------------------
# Boundary condition reduction (setBCs)
# ------------------------------------------------------------------
 
def set_bcs(
    Bglob: sp.csr_matrix,
    BC: np.ndarray,
    vnum: np.ndarray,
):
    """
    Impose the edge boundary conditions on the global strain–displacement
    matrix Bglob, exactly mirroring MATLAB’s *setBCs*.
 
    Parameters
    ----------
    Bglob : (m, n) scipy.sparse.csr_matrix
        Full global strain–displacement matrix.
    BC    : (2, 4) int ndarray
        Boundary‑condition flags (row 0 → ux, row 1 → uy; columns 1–4 are
        edges top/left/bottom/right).  Flags: 0 free, 1 fixed, 2 constant.
    vnum  : (ny+1, nx+1) int ndarray
        Grid of virtual‑node numbers.
 
    Returns
    -------
    Bbar         : csr_matrix
        Matrix with slave/fixed columns removed and slaves merged into
        their masters.
    active_dofs  : (k,) int ndarray
        Column indices that remain in *Bbar* (global numbering).
    free_dof     : (ℓ,) int ndarray
        DOFs that are not fixed, not slaves, **and not masters of a
        constant edge** – identical to MATLAB’s *freeDof*.
    """
    # ---- node lists for the four edges ----------------------------------
    edges = {
        1: vnum[0, :],     # top
        2: vnum[:, 0],     # left
        3: vnum[-1, :],    # bottom
        4: vnum[:, -1],    # right
    }
 
    slave2master: dict[int, int] = {}   # {slave_dof: master_dof}
    fixed: set[int] = set()             # fixed DOFs
 
    # ---- loop over edges ------------------------------------------------
    for e, nodes in edges.items():
 
        # master node: TL for edges 1&2 ; BR for edges 3&4  (MATLAB rule)
        master_node = nodes[0] if e in (1, 2) else nodes[-1]
        mx, my = 2 * master_node, 2 * master_node + 1
 
        x_dofs, y_dofs = 2 * nodes, 2 * nodes + 1
        ux_flag, uy_flag = BC[0, e - 1], BC[1, e - 1]
 
        # ---- x‑direction flags -----------------------------------------
        if ux_flag == 1:                       # fixed
            fixed.update(x_dofs)
        elif ux_flag == 2:                     # constant
            slave2master.update({s: mx for s in x_dofs[1:]})
 
        # ---- y‑direction flags -----------------------------------------
        if uy_flag == 1:
            fixed.update(y_dofs)
        elif uy_flag == 2:
            slave2master.update({s: my for s in y_dofs[1:]})
 
    # ---- conflict test --------------------------------------------------
    if any(m in fixed for m in slave2master.values()):
        raise ValueError("Incompatible BCs: a master DOF is also fixed")
 
    # ---- merge slave columns into their master --------------------------
    Bbar = Bglob.tolil(copy=True)
    for s, m in slave2master.items():
        Bbar[:, m] += Bbar[:, s]
 
    # ---- drop columns of fixed + slave DOFs -----------------------------
    drop = np.fromiter(fixed | slave2master.keys(), int)
    active_dofs = np.setdiff1d(np.arange(Bbar.shape[1]), drop)
    Bbar = Bbar[:, active_dofs].tocsr()
 
    # ---- MATLAB‑style freeDof list --------------------------------------
    masters_const = set(slave2master.values())          # masters of “constant” edges
    free_dof = np.array(
        sorted(
            set(range(Bglob.shape[1]))
            - fixed
            - masters_const
            - set(slave2master.keys())
        ),
        dtype=int,
    )
 
    return Bbar, active_dofs, free_dof
 
 
# ------------------------------------------------------------------
# Main driver – generates the virtual mesh and B‑matrices
# ------------------------------------------------------------------
 
def generate_virtual_mesh(test_data: dict, index_list: np.ndarray, opts: dict):
    """Python implementation of MATLAB *generateVirtualMesh* (GUI‑free)."""
    index_list = np.asarray(index_list)
    if index_list.dtype == bool:
        index_list = np.flatnonzero(index_list)
 
    X, Y = np.asarray(test_data["X"], dtype=float), np.asarray(test_data["Y"], dtype=float)
    mask = test_data["nullList"]
    X = X.copy(); Y = Y.copy()
    X[mask] = Y[mask] = np.nan
 
    n_x_elem, n_y_elem = map(int, opts["VFmeshSize"])
    nlgeom = bool(opts.get("nlgeom", 0))
 
    # ------------------------------------------------------------------
    # MATLAB flag: 1 if Y decreases downward (Cartesian); 0 otherwise
    # ------------------------------------------------------------------
    y_down_decrease_flag: bool = np.nanmean(np.diff(Y, axis=0)) <= 0
 
    # ------------------------------------------------------------------
    # Data‑element centroid grid (needed for node seeding)
    # ------------------------------------------------------------------
    mesh_elems_x, mesh_elems_y = compute_mesh_grids(X, Y)
 
    if not y_down_decrease_flag:  # flip if Y increases downward
        mesh_elems_x = np.flipud(mesh_elems_x)
        mesh_elems_y = np.flipud(mesh_elems_y)
 
    # ------------------------------------------------------------------
    # Virtual nodes (snapped)
    # ------------------------------------------------------------------
    Xvec, Yvec = seed_virtual_nodes(
        (n_x_elem, n_y_elem), mesh_elems_x, mesh_elems_y, y_down_decrease_flag
    )
    Xv, Yv = np.meshgrid(Xvec, Yvec)  # shape (n_y_elem+1, n_x_elem+1)
 
    # ------------------------------------------------------------------
    # Connectivity (MATLAB‑consistent orientation)
    # ------------------------------------------------------------------
    vnum = np.arange((n_y_elem + 1) * (n_x_elem + 1), dtype=int).reshape(
        (n_y_elem + 1, n_x_elem + 1)
    )
 
    vCoordsGrid = vnum.copy()
 
    conn = np.zeros((4, n_x_elem * n_y_elem), dtype=int)
    k = 0
    for j in range(n_y_elem):
        for i in range(n_x_elem):
            if y_down_decrease_flag:
                # Y decreases downward → use MATLAB branch 1 ordering
                conn[:, k] = [
                    vnum[j, i],
                    vnum[j + 1, i],
                    vnum[j + 1, i + 1],
                    vnum[j, i + 1],
                ]
            else:
                # Y increases downward → MATLAB branch 2 ordering
                conn[:, k] = [
                    vnum[j + 1, i],
                    vnum[j, i],
                    vnum[j, i + 1],
                    vnum[j + 1, i + 1],
                ]
            k += 1
 
    # ------------------------------------------------------------------
    # Associate each measurement point to a virtual element
    # ------------------------------------------------------------------
    xy_pts = np.c_[X.ravel()[index_list], Y.ravel()[index_list]]
    Xvf, Yvf = Xv.ravel(), Yv.ravel()
 
    pt_elem = -np.ones(xy_pts.shape[0], dtype=int)
    for e in range(conn.shape[1]):
        tol = 1e-12 * max(np.nanmax(X) - np.nanmin(X),
                  np.nanmax(Y) - np.nanmin(Y))   # characteristic length × tiny factor
        poly = Pathm(np.c_[Xvf[conn[:, e]], Yvf[conn[:, e]]])
        inside = poly.contains_points(xy_pts, radius=tol)   # ← edge points included
        pt_elem[inside] = e
 
    keep_mask = pt_elem >= 0
    if not np.all(keep_mask):
        print(
            f"[generate_virtual_mesh] WARNING: {(~keep_mask).sum()} "
            "measurement points lie outside the virtual mesh – they are ignored."
        )
        xy_pts = xy_pts[keep_mask]
        pt_elem = pt_elem[keep_mask]
        kept_idx = index_list[keep_mask]
    else:
        kept_idx = index_list
 
    n_meas_pt = xy_pts.shape[0]
    strain_rows_per_pt = 4 if nlgeom else 3
    n_rows_Bglob = strain_rows_per_pt * n_meas_pt
    n_dofs = 2 * Xv.size
 
    # ------------------------------------------------------------------
    # Assemble sparse global strain‑displacement matrix  Bglob
    # ------------------------------------------------------------------
    data, rows, cols = [], [], []
    dataN, rowsN, colsN = [], [], []  
    for p, e in enumerate(pt_elem):
        ln = conn[:, e]
        dofs = np.column_stack((2 * ln, 2 * ln + 1)).ravel()
        xi, eta = inverse_transform(np.c_[Xvf[ln], Yvf[ln]], xy_pts[p])
        N, dN_dxi = shape_functions(xi, eta)   # keep N for NGlob
        J = dN_dxi.T @ np.c_[Xvf[ln], Yvf[ln]]
        if abs(np.linalg.det(J)) < 1e-12:
            continue  # degenerate quad
        dNdx = la.solve(J.T, dN_dxi.T).T
        B = strain_matrix(dNdx, nlgeom)
        r0 = p * strain_rows_per_pt
        rr, cc = np.indices(B.shape)
        rows.extend((rr + r0).ravel())
        cols.extend(dofs[cc.ravel()])
        data.extend(B.ravel())
        # --- NGlob -----------------------------------------------------
        rowsN.extend([p] * 4)          # same point‑row four times
        colsN.extend(ln)               # the 4 node numbers of this element
        dataN.extend(N)                # the 4 shape‑function values
 
    Bglob = sp.csr_matrix((data, (rows, cols)), shape=(n_rows_Bglob, n_dofs))
 
    nNodes = Xv.size               # total virtual nodes
    NGlob  = sp.csr_matrix(
        (dataN, (rowsN, colsN)),
        shape=(n_meas_pt, nNodes),
    )

    # ------------------------------------------------------------------
    # Boundary conditions & pseudo‑inverse
    # ------------------------------------------------------------------
    Bbar, active_dofs,free_dof = set_bcs(Bglob, opts["BCsettings"], vnum)
    Binv = la.pinv(Bbar.toarray())
 
    return {
        "Bglob": Bglob,
        "Bbar": Bbar,
        "Binv": Binv,
        "Xv": Xv,
        "Yv": Yv,
        "connectivity": conn,
        "activeDOF": active_dofs,
        "ptElemAss": pt_elem,
        "YDownDecreaseFlag": y_down_decrease_flag,
        "keptIdx": kept_idx,
        "freeDof":    free_dof,
        "vCoordsGrid": vCoordsGrid,
        "NGlob":      NGlob,
        "BCsettings": opts["BCsettings"],   # handy for downstream code
    }
 
def generateData(x,y):
    nullList = np.isnan(x)
    test_data = dict(X=x, Y=y, nullList=nullList)
    idx = np.flatnonzero(~nullList)
    return test_data, idx

def sensitivity_vfs(
    refmap: np.ndarray,
    meshData: Dict,
    options: Dict,
) -> Tuple[Dict, np.ndarray]:
    """
    Python port of MATLAB *sensitivityVFs*.
 
    Parameters
    ----------
    refmap : (ny, nx, nComp, nSteps) ndarray
        Incremental stress‑sensitivity map.
        nComp = 3 (small strain) or 4 (NLGEOM on).
    meshData : dict
        Output from *generate_virtual_mesh* **augmented** with
        'BCsettings' (2×4 int array).
    options : dict  – must contain key 'nlgeom' (0 / 1).
 
    Returns
    -------
    VF : dict  with keys
        'eps' : (ny, nx, nComp, nSteps) ndarray  – virtual strains
        'u'   : (nSteps, 2, 4) ndarray           – mean edge V.U.
    VU : (ny, nx, 2, nSteps) ndarray
        Full‑field virtual displacements (for post‑processing).
    """
    # -------------------------------------------------------
    # Unpack mesh quantities
    # -------------------------------------------------------
    Bglob       = meshData["Bglob"]            # sparse
    Binv        = meshData["Binv"]             # dense  (k × m)
    act_dofs    = meshData["activeDOF"]        # (k,)
    BC          = meshData["BCsettings"]       # (2,4)
    Xv, Yv      = meshData["Xv"], meshData["Yv"]
    conn        = meshData["connectivity"]     # 4 × nElem
    pt_elem     = meshData["ptElemAss"]        # (nPts,)
    idx_meas    = meshData["keptIdx"]          # same length as pt_elem
    y_flag      = meshData["YDownDecreaseFlag"]
    Nglob      = meshData["NGlob"]
 
    ny_map, nx_map, nComp, nSteps = refmap.shape
    nPts       = idx_meas.size
    nDofs      = Bglob.shape[1]
    nNodes     = Xv.size
 
    # Build vCoordsGrid (node‑number grid) for edge bookkeeping
    vCoordsGrid = np.arange(nNodes, dtype=int).reshape(Xv.shape)
 
    # -------------------------------------------------------
    # Pre‑allocate results
    # -------------------------------------------------------
    VF_eps = np.zeros((ny_map, nx_map, nComp, nSteps))
    VF_eps[:] = np.nan
    VF_u   = np.zeros((nSteps, 2, 4))
    VU     = np.zeros((ny_map, nx_map, 2, nSteps))
    VU[:]  = np.nan
 
    # -------------------------------------------------------
    # Time‑loop
    # -------------------------------------------------------
    for step in range(nSteps):
 
        # ---- build right‑hand vector  b  -------------------
        if options["nlgeom"]:
            Exx = refmap[:, :, 0, step].flat[idx_meas]
            Eyy = refmap[:, :, 1, step].flat[idx_meas]
            Eyx = refmap[:, :, 2, step].flat[idx_meas]
            Exy = refmap[:, :, 3, step].flat[idx_meas]
            #b = np.concatenate([Exx, Eyy, Exy, Eyx])
            b = np.zeros(4*Exx.size)
            b[0::4] = Exx
            b[1::4] = Eyy
            b[2::4] = Eyx
            b[3::4] = Exy
            
        else:
            Exx = refmap[:, :, 0, step].flat[idx_meas]
            Eyy = refmap[:, :, 1, step].flat[idx_meas]
            Exy = refmap[:, :, 2, step].flat[idx_meas]
            #b = np.concatenate([Exx, Eyy, Exy])
            #print('Here')
            b = np.zeros(3*Exx.size)
            b[0::3] = Exx
            b[1::3] = Eyy
            b[2::3] = Exy
 
        # ---- solve for virtual node displacements ----------
        vU = np.zeros(nDofs)
        vU[act_dofs] = Binv @ b                 # least‑squares solution

        ve = Bglob @ vU

        # ---- enforce boundary conditions -------------------
        vU = apply_bcs(vU, BC, vCoordsGrid)
 
        # ---- project to strain field  ve -------------------
        ve = Bglob @ vU
        #print(ve.shape)
        # ---- store virtual strains -------------------------
        if options["nlgeom"]:
            VF_eps[:, :, 0, step].flat[idx_meas] = ve[0::4]   # ε_xx
            VF_eps[:, :, 1, step].flat[idx_meas] = ve[1::4]   # ε_yy
            VF_eps[:, :, 3, step].flat[idx_meas] = ve[3::4]   # ε_xy
            VF_eps[:, :, 2, step].flat[idx_meas] = ve[2::4]   # ε_yx
        else:
            VF_eps[:, :, 0, step].flat[idx_meas] = ve[0::3]   # ε_xx
            VF_eps[:, :, 1, step].flat[idx_meas] = ve[1::3]   # ε_yy
            VF_eps[:, :, 2, step].flat[idx_meas] = ve[2::3]   # ε_xy
 
         
        vU_all_X = Nglob*vU[0::2]
        vU_all_Y = Nglob*vU[1::2]

        VU[:, :, 0, step].flat[idx_meas] = vU_all_X
        VU[:, :, 1, step].flat[idx_meas] = vU_all_Y
 
        # ---- mean virtual disp. along the four edges --------
        # Edge numbering: 1 top, 2 left, 3 bottom, 4 right
        edges = (
            vCoordsGrid[0, :],             # 1
            vCoordsGrid[:, 0],             # 2
            vCoordsGrid[-1, :],            # 3
            vCoordsGrid[:, -1],            # 4
        )
        for e, nodes in enumerate(edges):
            VF_u[step, 0, e] = vU[2 * nodes].mean()       # ux
            sign = 1 if y_flag else -1
            VF_u[step, 1, e] = sign * vU[2 * nodes + 1].mean()  # uy
 
    # ---- build output dict ---------------------------------------------
    VF = {"eps": VF_eps, "u": VF_u}
    return VF, VU
 
 
# -----------------------------------------------------------------------
# Helper: apply boundary conditions (master–slave) – direct MATLAB port
# -----------------------------------------------------------------------
def apply_bcs(
    vU: np.ndarray,
    BC: np.ndarray,
    vGrid: np.ndarray,
) -> np.ndarray:
    """Impose the virtual‑displacement BCs in‑place and return *vU*."""
    for edge in range(1, 5):
        if edge == 1:                           # top
            nodes = vGrid[0, :]
        elif edge == 2:                         # left
            nodes = vGrid[:, 0]
        elif edge == 3:                         # bottom
            nodes = vGrid[-1, :]
        else:                                   # right
            nodes = vGrid[:, -1]
 
        master = nodes[0] if edge in (1, 2) else nodes[-1]
        x_dofs = 2 * nodes
        y_dofs = x_dofs + 1
        mx, my = 2 * master, 2 * master + 1
 
        # ---- x ----------------------------------------------------------
        if   BC[0, edge - 1] == 1:             # fixed
            vU[x_dofs] = 0.0
        elif BC[0, edge - 1] == 2:             # constant
            vU[x_dofs[1:]] = vU[mx]
 
        # ---- y ----------------------------------------------------------
        if   BC[1, edge - 1] == 1:
            vU[y_dofs] = 0.0
        elif BC[1, edge - 1] == 2:
            vU[y_dofs[1:]] = vU[my]
 
    return vU