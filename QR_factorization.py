import numpy as np
from decimal import Decimal as dec
dd = float
"""
want to implent QR. First need householder reflector.
Get first row. Make reflection vector. Make reflector. 
"""





def HH_vector(v : np.array) :
    v_height = np.shape(v)[0]
    if np.shape(v)[1] != 1 :
        raise TypeError ("Not a vector")

    # ensure the vector uses floating-point dtype so assignments keep decimals
    v = v.astype(dd)

    # don't attempt to reflect the zero vector
    if [v[k][0] for k in range(v_height)] == [0.0 for k in range(v_height)] :
        v = np.array([[1] if k == 0 else [0] for k in range(v_height)])

    # compute the 2-norm of v
    norm = np.linalg.norm(v)

    # choose sign to avoid cancellation: v <- v + sign(v0)*||v|| e1
    sign = 1.0 if v[0][0] >= 0.0 else -1.0
    v[0][0] = v[0][0] + sign*norm

    # compute norm of the modified vector
    res_norm = np.linalg.norm(v)

    # guard against division by (near) zero due to numerical cancellation
    eps = np.finfo(float).eps
    if res_norm <= eps:
        # if the reflector vector collapsed to (near) zero, return the
        # canonical first basis vector as a unit reflector direction.
        unit_res = [[1.0 if i == 0 else 0.0] for i in range(v_height)]
        return np.array(unit_res, dtype=dd)

    # normalize and return as column vector
    unit_res = [[v[i][0] / res_norm] for i in range(v_height)]
    return np.array(unit_res, dtype=dd)

def HH_reflector(v) : 
    axis = HH_vector(v)
    vv_star = axis@axis.T
    dimn = np.shape(vv_star)[0]
    id = np.identity(dimn)
    return id - 2*vv_star





def Id_block_mtx(M,n) :
    r,c = np.shape(M)
    if r != c :
        raise TypeError
    if n == 0 :
        return M
    aii = np.identity(n,dtype=dd)
    id_dimn = n - r
    # use floats for the identity/cushion matrix so later multiplications
    # and reflector placements preserve fractional values
    cush = id_dimn
    i = cush
    j = cush
    while i < n :
        while j < n :
            aii[i,j] = M[i-cush, j-cush]
            j+=1
        j = id_dimn
        i += 1
    return aii


def nth_col(n, M : np.array) :
    # return a column vector with float dtype to avoid integer casting
    return np.array([[M[k][n]] for k in range(np.shape(M)[0])], dtype=dd)

def vec_starting(v : np.array, n : int) :
    r,c = np.shape(v)
    if c != 1 :
        raise TypeError (f"Not a vector, has shape {(r,c)}")
    if n >= r :
        raise IndexError (f"Cannot access {n}-index entry of a vector with {r} entries")
    return np.array(v[n:])


def QR(M) :
    m_rows, m_cols = np.shape(M)
    res = M
    i = 1
    j = 1
    # print(f"M has shape {np.shape(M)}")
    first_guy = HH_reflector(nth_col(0, M))
    # print(f"First reflector has shape {np.shape(first_guy)}")
    # print()
    # print(f"First reflector is \n{first_guy}")
    res = HH_reflector(nth_col(0, M))@M #REPLACED mul HERE
    My_Q = first_guy
    while i < m_rows-1 and j < m_cols-1 :
        prep_vector = nth_col(j, res)
        # print(f"Right now, prep_vector is \n{prep_vector}")
        prep_vector = vec_starting(prep_vector, i)
        # print(f"Newly edited prep_vector is \n{prep_vector}")
        Q = HH_reflector(prep_vector)
        # print()
        # print(f"My Small HH_ref is \n{Q}, \n i is {i}")
        size_corrected_Q = Id_block_mtx(Q, m_rows)
        # print(f"res has shape {np.shape(res)}, and size_corrected_Q has shape {np.shape(size_corrected_Q)}")
        res = size_corrected_Q@res
        My_Q = size_corrected_Q@My_Q
        i += 1
        j += 1
    return My_Q.T, res #QR = M




def UHess(M) :
    m_rows, m_cols = np.shape(M)
    res = M.copy()
    i = 2
    j = 2
    # print(f"M has shape {np.shape(M)}")
    first_guy = HH_reflector(vec_starting(nth_col(0, M),1))
    # print(f"First reflector has shape {np.shape(first_guy)}")
    # print()
    # print(f"First reflector is \n{first_guy}")
    My_Q = Id_block_mtx(first_guy, m_rows)
    # print(f"First Q being used is \n{My_Q}")
    # print(f"res is \n{res}")
    # print("Now we will find QresQt")
    res = My_Q@res@My_Q.T
    while i < m_rows-1 and j < m_cols-1 :
        prep_vector = nth_col(j, res)
        # print(f"Right now, prep_vector is \n{prep_vector}")
        prep_vector = vec_starting(prep_vector, i)
        # print(f"Newly edited prep_vector is \n{prep_vector}")
        Q = HH_reflector(prep_vector)
        # print()
        # print(f"My Small HH_ref is \n{Q}, \n i is {i}")
        size_corrected_Q = Id_block_mtx(Q, m_rows)

        # print(f"res has shape {np.shape(res)}, and size_corrected_Q has shape {np.shape(size_corrected_Q)}. My_Q has size {np.shape(My_Q)}")
        res = size_corrected_Q@res@size_corrected_Q.T
        My_Q = My_Q@size_corrected_Q
        i += 1
        j += 1
    return res, My_Q # QRQ = M