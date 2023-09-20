import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple fast_viterbi(str t, str ref,
                       np.ndarray[DTYPE_t, ndim=1] delta, 
                       np.ndarray[DTYPE_t, ndim=1] tau,
                       np.ndarray[DTYPE_t, ndim=1] MM,
                       np.ndarray[DTYPE_t, ndim=1] epsilon, 
                       DTYPE_t rho,
                       np.ndarray[DTYPE_t, ndim=1] DM,
                       np.ndarray[DTYPE_t, ndim=1] gamma, 
                       DTYPE_t pi,
                       np.ndarray[DTYPE_t, ndim=1] IM, 
                       DTYPE_t q, DTYPE_t match, DTYPE_t mismatch):
    cdef int n = len(ref)
    cdef int m = len(t)
    cdef int i, j
    cdef DTYPE_t[:, :] M = cython.view.array(shape=(n+1, m+1), itemsize=sizeof(DTYPE_t), format="d")
    cdef DTYPE_t[:, :] D = cython.view.array(shape=(n+1, m+1), itemsize=sizeof(DTYPE_t), format="d")
    cdef DTYPE_t[:, :] I = cython.view.array(shape=(n+1, m+1), itemsize=sizeof(DTYPE_t), format="d")
    cdef ITYPE_t[:,:,:] R = np.full((n+1, m+1, 3), -1, dtype=np.int32)

    # Initialize the matrix
    for i in range(n+1):
        for j in range(m+1):
            if i == 0 and j == 0:
                M[i, j] = 0
                D[i, j] = -99999
                I[i, j] = -99999
            elif i == 0:
                M[i, j] = -99999
                D[i, j] = -99999
                I[i, j] = tau[0] + q + (gamma[0] + q) * (j-1)
                R[i,j,1] = 1
                R[i,j,2] = 2
            elif j == 0:
                M[i, j] = -99999
                D[i, j] = delta[0] + q + sum([epsilon[k] + q for k in range(1,i)])
                I[i, j] = -99999
                R[i,j,2] = 2
            else:
                M[i, j] = -99999
                D[i, j] = -99999
                I[i, j] = -99999

    for i in range(1, n+1):
        for j in range(1, m+1):
            M_max, M_idx = -99999, 0
            D_max, D_idx = -99999, 0
            I_max, I_idx = -99999, 0

            for k in range(3):
                M_val = [MM[i-1], DM[i-1], IM[i-1]][k] + [M[i-1, j-1], D[i-1, j-1], I[i-1, j-1]][k]
                if M_val > M_max:
                    M_max, M_idx = M_val, k

                D_val = [delta[i-1], epsilon[i-1], pi][k] + [M[i-1, j], D[i-1, j], I[i-1, j]][k]
                if D_val > D_max:
                    D_max, D_idx = D_val, k

                I_val = [tau[i], rho, gamma[i]][k] + [M[i, j-1], D[i, j-1], I[i, j-1]][k]
                if I_val > I_max:
                    I_max, I_idx = I_val, k

            M[i, j] = (match * (ref[i-1] == t[j-1]) + mismatch * (ref[i-1] != t[j-1])) + M_max
            D[i, j] = q + D_max
            I[i, j] = q + I_max

            R[i, j, 0] = M_idx
            R[i, j, 1] = D_idx
            R[i, j, 2] = I_idx

    cdef int state    
    R[1, 1, 0] = -1
    R[0, 1, 1] = -1
    R[0, 1, 2] = -1
    R[1, 0, 2] = -1
    scores = [M[n, m] + MM[n],
              D[n, m],
              I[n, m] + IM[n]]
    best_score = max(scores)
    state = scores.index(best_score)
    cdef list aligned_ref = []
    cdef list aligned_t = []
    i = n
    j = m
    while (state != -1):
        if state == 0:
            state = R[i, j, 0]
            aligned_ref.append(ref[i-1])
            aligned_t.append(t[j-1])
            i -= 1
            j -= 1

        elif state == 1:
            state = R[i, j, 1]
            aligned_ref.append(ref[i-1])
            aligned_t.append("-")
            i -= 1

        elif state == 2:
            state = R[i, j, 2]
            aligned_ref.append("-")
            aligned_t.append(t[j-1])
            j -= 1
    
    if i > 0 and j > 0:
        raise RuntimeError("Viterbi algorithm failed for sequence %s" %t)
    ref = ref[:i] + '-'*j + ''.join(reversed(aligned_ref))
    t = '-'*i + t[:j] + ''.join(reversed(aligned_t))
    return ref,t,best_score
