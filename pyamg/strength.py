"""Strength of Connection functions.

Requirements for the strength matrix C are:
    1) Nonzero diagonal whenever A has a nonzero diagonal
    2) Non-negative entries (float or bool) in [0,1]
    3) Large entries denoting stronger connections
    4) C denotes nodal connections, i.e., if A is an nxn BSR matrix with
       row block size of m, then C is (n/m) x (n/m)

"""

from warnings import warn

import numpy as np
from scipy import sparse
from scipy.sparse import csr_array
from . import amg_core
from .relaxation.relaxation import jacobi
from .util.linalg import approximate_spectral_radius
from .util.utils import (scale_rows_by_largest_entry, amalgamate, scale_rows,
                         get_block_diag, scale_columns)
from .util.params import set_tol



def compute_mu(aii,ajj,aij,aji,si,sj,reciprocal = 1):
  
  if (aii == 0) or (ajj==0):
    return 0
  elif (aii + ajj - si - sj) == 0:
    return 0
  b = ( aii * ajj ) / ( aii + ajj )
  c = (( aii - si )*( ajj - sj )) / ( aii + ajj - si - sj )
  d = ( aji + aij )/2

  if ( (c-d) == 0 ) or (b == 0):
    return 0

  if reciprocal:
    return ( c - d ) / ( 2 * b )

  return 1 - (( 2 * b ) / ( c - d ))

def get_U(A,theta):
  U = []

  for i in range(A.shape[0]):
    sm = 0
    for j in range(A.shape[0]):
      if j != i:
        sm += abs(A[i,j]+A[j,i])/2
    if (theta*sm) < A[i,i]:
      U.append(i)

  return U

def compute_Us(A, theta):
  index_type = 'd'

  s = np.empty(A.shape[0], dtype=index_type)
  diag_indices = [0 for i in range(A.shape[0])]
  
  ''' sums exclude the diagonal entries '''
  rowsum = [0 for i in range(A.shape[0])]
  colsum = [0 for i in range(A.shape[0])]
  absrowsum = [0 for i in range(A.shape[0])]
  abscolsum = [0 for i in range(A.shape[0])]
  U = []
  notU = []
  for i in range(A.shape[0]):
    row_start = A.indptr[i]
    row_end = A.indptr[i+1]
    columns = A.indices[row_start:row_end]
    for k in range(row_start,row_end):
      j = A.indices[k]
      e = A.data[k]
      abse = abs(e)
      if i != j:
        rowsum[i] += e
        colsum[j] += e
        absrowsum[i] += abse
        abscolsum[j] += abse
      else:
        diag_indices[i] = k
  
  for i in range( A.shape[0] ):
    s[i] = (rowsum[i] + colsum[i])
    s[i] = -s[i]/2
    sm = (absrowsum[i] + abscolsum[i]) / 2

    if( A.indices[diag_indices[i]] == i ) and ( A.data[diag_indices[i]] < ( theta * sm ) ):
      U.append(i)
    else:
      notU.append(i)

  return U,s,diag_indices,notU
'''
  for i in range(A.shape[0]):
    s[i] = 0
    for j in range(A.shape[0]):
      if j != i:
        #s[i] += (A[i,j] +A[j,i])/2
        s[i] += (get_csr_elem(A,i,j) + get_csr_elem(A,j,i))/2
    s[i] *= -1

  return s
'''
def get_csr_elem(A,i,j):
    row_start = A.indptr[i]
    row_end = A.indptr[i+1]
    columns = A.indices[row_start:row_end]
    if j not in columns:
      return 0

    it =  range(row_start,row_end)
    if i > j:
      it = reversed(it)
    for k in it:
      if A.indices[k] == j:
        return A.data[k]

def pairwise_soc(A,U,s,D, notU, replacezeros = 0,smooth=0,reciprocal=1):
  '''
  soc should have sparsity of A so initialize with mu=A
  '''
  mu = A.copy()
  undefined_entry_count = 0

  ''' zero out entries we won't compute '''
  for i in notU:
    row_starti = mu.indptr[i]
    row_endi = mu.indptr[i+1]
    for k in range(row_starti, row_endi):
      mu.data[k] = 0

  for i in U:
  #for i in range(A.shape[0]):
    row_starti = A.indptr[i]
    row_endi = A.indptr[i+1]

    for k in range(row_starti,D[i]):

      if mu.indices[k] in notU:
        mu.data[k] = 0

    aii = A.data[D[i]]
    mu.data[D[i]] = 1
    si = s[i]
    '''
    this loop is unrolled
    we find mu(i,j) and mu(j,i)
    '''
    for kupper in range(D[i]+1, row_endi):
      
      j = A.indices[kupper]
      sj = s[j]

      row_startj = A.indptr[j]
      row_endj = A.indptr[j+1]

      '''find index of aji'''      
      klower = row_startj
      '''note if D[j] is zero then ajj is not present in A'''
      while (A.indices[klower] < i) and (klower < D[j] ):
        klower += 1
      
      ''' aji = aji or 0 '''
      mulower = mu.data[klower]
      resetlower = 0
      if A.indices[klower] == i:
        aji = A.data[klower]
        ajj = A.data[D[j]]
        mu.data[D[j]] = 1

      else:
        aji = 0
        ajj = 0
        resetlower = 1
        

      aij = A.data[kupper]

      ''' condition from the pairwise aggregation paper '''
      mu_nonzero =  (aji != 0) and (aij!=0)
      
      mu_nonzero = mu_nonzero and (aii + ajj - si - sj >= 0)

      mu_k = compute_mu( aii, ajj, aij, aji, si,sj , reciprocal=reciprocal)

      ssum = abs(si)+abs(sj)
      if mu_nonzero and ( mu_k != 0 ):

        #mu.data[kupper] = compute_mu(aii, ajj, aij, aji, s[i],s[j] )
      
        #mu.data[klower] = compute_mu(ajj, aii, aji, aij, s[j],s[i] )
        mu.data[kupper] = abs( mu_k /ssum)
        mu.data[klower] = abs( mu_k /ssum)

      elif smooth:
        undefined_entry_count += 1
        mu.data[kupper] = abs( mu.data[kupper] + mu_k ) / 2
        mu.data[klower] = abs( mu.data[klower] + mu_k ) / 2

      elif replacezeros:
        undefined_entry_count += 1
        mu.data[kupper] = abs( mu.data[kupper] / si)
        mu.data[klower] = abs( mu.data[klower] / sj)
      
      else:
        undefined_entry_count += 1
        mu.data[kupper] = 0
        mu.data[klower] = 0

      if j in notU:
        mu.data[kupper] = 0

      ''' reset value if klower doesn't point to elem j,i '''
      if resetlower:
        mu.data[klower] = mulower

      if mu.indices[klower] in notU:
        mu.data[klower] = 0

  if undefined_entry_count > A.indptr[A.shape[0] ]/2: 
    warn('Pairwise SOC has > 50% undefined entries', sparse.SparseEfficiencyWarning)
  return mu
      

def pairwise_soc1(A,s,mu,theta, maximize=1, reciprocal=1, allentries=1):
  U = get_U(A,theta)

  #for i in U:
  for i in range(A.shape[0]):
    jopt = i
    muopt = 0
    #for j in U:
    for j in range(A.shape[0]):
      if (i!=j) and (A[i,j]!= 0) and ((A[i,i] + A[j,j] - s[i] -s[j])>=0):
        aii = A[i,i]
        ajj = A[j,j]

        jmu = compute_mu(aii, ajj, A[i,j], A[j,i], s[i],s[j], reciprocal)
        if (jmu > 0) and allentries:
          mu[i,j] = jmu
        elif maximize and (jmu > muopt):
          muopt = jmu
          jopt = j
        elif not maximize and (jmu < jopt):
          muopt = jmu
          jopt = j
    if muopt > 0:
      mu[i,jopt] = muopt

  return sparse.csr_matrix(mu)

'''
in the paper small values of mu indicate high degree of connectedness
this library assumes strength of connection matrices use large values to indicate high degree of connectedness

entries in returned matrix are reciprocal of mu as defined in algorithm 4.2 from the pairwise aggregation paper
'''
def pairwise_strength_of_connection(A, theta=0.5, reciprocal=1,replacezeros=0,smooth=0):
  if A.format != 'csr':
    A = csr_array(A)
  A.sort_indices()
  U, s, D, notU = compute_Us( A, theta )
  return pairwise_soc(A,U,s,D,notU, replacezeros=replacezeros,smooth=smooth,reciprocal=reciprocal)



def distance_strength_of_connection(A, V, theta=2.0, relative_drop=True):
    """Distance based strength-of-connection.

    Parameters
    ----------
    A : csr_array or bsr_array
        Square, sparse matrix in CSR or BSR format.
    V : array
        Coordinates of the vertices of the graph of `A`.
    theta : float
        Drop tolerance (distance).
    relative_drop : bool
        If false, then a connection must be within a distance of theta
        from a point to be strongly connected.
        If true, then the closest connection is always strong, and other points
        must be within theta times the smallest distance to be strong.

    Returns
    -------
    csr_array
        `C(i,j) = distance(point_i, point_j)`
        Strength of connection matrix where strength values are
        distances, i.e. the smaller the value, the stronger the connection.
        Sparsity pattern of `C` is copied from `A`.

    Notes
    -----
    - `theta` is a drop tolerance that is applied row-wise
    - If a BSR matrix given, then the return matrix is still CSR.  The strength
      is given between super nodes based on the BSR block size.

    Examples
    --------
    >>> from pyamg.gallery import load_example
    >>> from pyamg.strength import distance_strength_of_connection
    >>> data = load_example('airfoil')
    >>> A = data['A'].tocsr()
    >>> vertices = data['vertices']
    >>> S = distance_strength_of_connection(A, vertices)

    """
    # Amalgamate for the supernode case
    if sparse.issparse(A) and A.format == 'bsr':
        sn = int(A.shape[0] / A.blocksize[0])
        u = np.ones((A.data.shape[0],))
        A = sparse.csr_array((u, A.indices, A.indptr), shape=(sn, sn))

    if not sparse.issparse(A) or A.format != 'csr':
        warn('Implicit conversion of A to csr', sparse.SparseEfficiencyWarning)
        A = sparse.csr_array(A)

    dim = V.shape[1]

    # Create two arrays for differencing the different coordinates such
    # that C(i,j) = distance(point_i, point_j)
    cols = A.indices
    rows = np.repeat(np.arange(A.shape[0], dtype=cols.dtype), A.indptr[1:] - A.indptr[0:-1])

    # Insert difference for each coordinate into C
    C = (V[rows, 0] - V[cols, 0])**2
    for d in range(1, dim):
        C += (V[rows, d] - V[cols, d])**2
    C = np.sqrt(C)
    C[C < 1e-6] = 1e-6

    C = sparse.csr_array((C, A.indices.copy(), A.indptr.copy()),
                          shape=A.shape)

    # Apply drop tolerance
    if relative_drop is True:
        if theta != np.inf:
            amg_core.apply_distance_filter(C.shape[0], theta, C.indptr,
                                           C.indices, C.data)
    else:
        amg_core.apply_absolute_distance_filter(C.shape[0], theta, C.indptr,
                                                C.indices, C.data)
    C.eliminate_zeros()

    C = C + sparse.eye_array(C.shape[0], C.shape[1], format='csr')

    # Standardized strength values require small values be weak and large
    # values be strong.  So, we invert the distances.
    C.data = 1.0 / C.data

    # Scale C by the largest magnitude entry in each row
    C = scale_rows_by_largest_entry(C)

    return C


def classical_strength_of_connection(A, theta=0.1, block=True, norm='abs'):
    """Classical strength of connection measure.

    Return a strength of connection matrix using the classical AMG measure
    An off-diagonal entry ``A[i,j]`` is a strong connection iff::

             |A[i,j]| >= theta * max(|A[i,k]|), where k != i     (norm='abs')
             -A[i,j]  >= theta * max(-A[i,k]),  where k != i     (norm='min')

    Parameters
    ----------
    A : csr_array or bsr_array
        Square, sparse matrix in CSR or BSR format.
    theta : float
        Threshold parameter in [0,1].
    block : bool, default True
        Compute strength of connection block-wise.
    norm : str, default 'abs'
        Measure used in computing the strength::

            'abs' : |C[i,j]| >= theta * max(|C[i,k]|), where k != i
            'min' : -C[i,j]  >= theta * max(-C[i,k]),  where k != i

        where C = A for non-block-wise computations.  For block-wise::

            'abs'  : C[i, j] is the maximum absolute value in block A[i, j]
            'min'  : C[i, j] is the minimum (negative) value in block A[i, j]
            'fro'  : C[i, j] is the Frobenius norm of block A[i, j]

    Returns
    -------
    csr_array
        Matrix graph defining strong connections.  `S[i,j] ~ 1.0` if vertex `i`
        is strongly influenced by vertex `j`, or block `i` is strongly influenced
        by block `j` if `block=True`.

    See Also
    --------
    symmetric_strength_of_connection : Symmetric measure used in SA.
    evolution_strength_of_connection : Relaxation based strength measure.

    Notes
    -----
    - A symmetric `A` does not necessarily yield a symmetric strength matrix `S`
    - Calls C++ function classical_strength_of_connection
    - The version as implemented is designed for M-matrices.  Trottenberg et
      al. use max `A[i,k]` over all negative entries, which is the same.  A
      positive edge weight never indicates a strong connection.
    - See [0]_ and [1]_

    References
    ----------
    .. [0] Briggs, W. L., Henson, V. E., McCormick, S. F., "A multigrid
        tutorial", Second edition. Society for Industrial and Applied
        Mathematics (SIAM), Philadelphia, PA, 2000. xii+193 pp.

    .. [1] Trottenberg, U., Oosterlee, C. W., Schuller, A., "Multigrid",
        Academic Press, Inc., San Diego, CA, 2001. xvi+631 pp.

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.strength import classical_strength_of_connection
    >>> n=3
    >>> stencil = np.array([[-1.0,-1.0,-1.0],
    ...                        [-1.0, 8.0,-1.0],
    ...                        [-1.0,-1.0,-1.0]])
    >>> A = stencil_grid(stencil, (n,n), format='csr')
    >>> S = classical_strength_of_connection(A, 0.0)

    """
    if sparse.issparse(A) and A.format == 'bsr':
        if (A.blocksize[0] != A.blocksize[1]) or (A.blocksize[0] < 1):
            raise ValueError('Matrix must have square blocks')
        blocksize = A.blocksize[0]
    else:
        blocksize = 1

    if (theta < 0 or theta > 1):
        raise ValueError('expected theta in [0,1]')

    # Block structure considered before computing SOC
    if block and sparse.issparse(A) and A.format == 'bsr':
        N = int(A.shape[0] / blocksize)

        # SOC based on maximum absolute value element in each block
        if norm == 'abs':
            data = np.max(np.max(np.abs(A.data), axis=1), axis=1)
        # SOC based on hard minimum of entry in each off-diagonal block
        elif norm == 'min':
            data = np.min(np.min(A.data, axis=1), axis=1)
        # SOC based on Frobenius norms of blocks
        elif norm == 'fro':
            data = np.conjugate(A.data) * A.data
            data = np.sum(np.sum(data, axis=1), axis=1)
        else:
            raise ValueError('Invalid choice of norm.')

        # drop small numbers
        data[np.abs(data) < 1e-16] = 0.0
    else:
        if not sparse.issparse(A) or A.format != 'csr':
            warn('Implicit conversion of A to csr', sparse.SparseEfficiencyWarning)
            A = sparse.csr_array(A)
        data = A.data
        N = A.shape[0]

    Sp = np.empty_like(A.indptr)
    Sj = np.empty_like(A.indices)
    Sx = np.empty_like(data)

    if norm in ('abs', 'fro'):
        amg_core.classical_strength_of_connection_abs(
            N, theta, A.indptr, A.indices, data, Sp, Sj, Sx)
    elif norm == 'min':
        amg_core.classical_strength_of_connection_min(
            N, theta, A.indptr, A.indices, data, Sp, Sj, Sx)
    else:
        raise ValueError('Unrecognized option for norm for strength.')

    S = sparse.csr_array((Sx, Sj, Sp), shape=[N, N])

    # Take magnitude and scale by largest entry
    S.data = np.abs(S.data)
    S = scale_rows_by_largest_entry(S)
    S.eliminate_zeros()

    if blocksize > 1 and not block:
        S = amalgamate(S, blocksize)

    return S


def symmetric_strength_of_connection(A, theta=0):
    """Symmetric Strength Measure.

    Compute strength of connection matrix using the standard symmetric measure

    An off-diagonal connection A[i,j] is strong iff::

        abs(A[i,j]) >= theta * sqrt( abs(A[i,i]) * abs(A[j,j]) )

    Parameters
    ----------
    A : csr_array
        Matrix graph defined in sparse format.  Entry `A[i,j]` describes the
        strength of edge `[i,j]`.
    theta : float
        Threshold parameter (positive).

    Returns
    -------
    csr_array
        Matrix graph defining strong connections.  `S[i,j]=1` if vertex `i`
        is strongly influenced by vertex `j`.

    See Also
    --------
    symmetric_strength_of_connection : Symmetric measure used in SA.
    evolution_strength_of_connection : Relaxation based strength measure.

    Notes
    -----
        - For vector problems, standard strength measures may produce
          undesirable aggregates.  A "block approach" from Vanek et al. is used
          to replace vertex comparisons with block-type comparisons.  A
          connection between nodes `i` and `j` in the block case is strong if::

          ||AB[i,j]|| >= theta * sqrt( ||AB[i,i]||*||AB[j,j]|| ) where AB[k,l]

          is the matrix block (degrees of freedom) associated with nodes `k` and
          l and ||.|| is a matrix norm, such a Frobenius.

        - See [1]_ for more details.

    References
    ----------
    .. [1] Vanek, P. and Mandel, J. and Brezina, M.,
       "Algebraic Multigrid by Smoothed Aggregation for
       Second and Fourth Order Elliptic Problems",
       Computing, vol. 56, no. 3, pp. 179--196, 1996.
       http://citeseer.ist.psu.edu/vanek96algebraic.html

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.strength import symmetric_strength_of_connection
    >>> n=3
    >>> stencil = np.array([[-1.0,-1.0,-1.0],
    ...                        [-1.0, 8.0,-1.0],
    ...                        [-1.0,-1.0,-1.0]])
    >>> A = stencil_grid(stencil, (n,n), format='csr')
    >>> S = symmetric_strength_of_connection(A, 0.0)

    """
    if theta < 0:
        raise ValueError('expected a positive theta')

    if sparse.issparse(A) and A.format == 'csr':
        # if theta == 0:
        #     return A

        Sp = np.empty_like(A.indptr)
        Sj = np.empty_like(A.indices)
        Sx = np.empty_like(A.data)

        fn = amg_core.symmetric_strength_of_connection
        fn(A.shape[0], theta, A.indptr, A.indices, A.data, Sp, Sj, Sx)

        S = sparse.csr_array((Sx, Sj, Sp), shape=A.shape)

    elif sparse.issparse(A) and A.format == 'bsr':
        M, N = A.shape
        R, C = A.blocksize

        if R != C:
            raise ValueError('matrix must have square blocks')

        if theta == 0:
            data = np.ones(len(A.indices), dtype=A.dtype)
            S = sparse.csr_array((data, A.indices.copy(), A.indptr.copy()),
                                  shape=(int(M / R), int(N / C)))
        else:
            # the strength of connection matrix is based on the
            # Frobenius norms of the blocks
            data = (np.conjugate(A.data) * A.data).reshape(-1, R * C)
            data = np.sqrt(data.sum(axis=1))
            A = sparse.csr_array((data, A.indices, A.indptr),
                                  shape=(int(M / R), int(N / C)))
            return symmetric_strength_of_connection(A, theta)
    else:
        raise TypeError('expected CSR or BSR sparse format')

    # Strength represents "distance", so take the magnitude
    S.data = np.abs(S.data)

    # Scale S by the largest magnitude entry in each row
    S = scale_rows_by_largest_entry(S)

    return S


def energy_based_strength_of_connection(A, theta=0.0, k=2):
    """Energy Strength Measure.

    Compute a strength of connection matrix using an energy-based measure.

    Parameters
    ----------
    A : sparse-matrix
        Matrix from which to generate strength of connection information.
    theta : float
        Threshold parameter in [0,1].
    k : int
        Number of relaxation steps used to generate strength information.

    Returns
    -------
    csr_array
        Matrix graph `S` defining strong connections.  The sparsity pattern
        of `S` matches that of A.  For BSR matrices, `S` is a reduced strength
        of connection matrix that describes connections between supernodes.

    Notes
    -----
    This method relaxes with weighted-Jacobi in order to approximate the
    matrix inverse.  A normalized change of energy is then used to define
    point-wise strength of connection values.  Specifically, let `v` be the
    approximation to the `i`-th column of the inverse, then

    (S_ij)^2 = <v_j, v_j>_A / <v, v>_A,

    where `v_j = v`, such that entry `j` in `v` has been zeroed out.  As is common,
    larger values imply a stronger connection.

    Current implementation is a very slow pure-python implementation for
    experimental purposes, only.

    See [1]_ for more details.

    References
    ----------
    .. [1] Brannick, Brezina, MacLachlan, Manteuffel, McCormick.
       "An Energy-Based AMG Coarsening Strategy",
       Numerical Linear Algebra with Applications,
       vol. 13, pp. 133-148, 2006.

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.strength import energy_based_strength_of_connection
    >>> n=3
    >>> stencil =  np.array([[-1.0,-1.0,-1.0],
    ...                      [-1.0, 8.0,-1.0],
    ...                      [-1.0,-1.0,-1.0]])
    >>> A = stencil_grid(stencil, (n,n), format='csr')
    >>> S = energy_based_strength_of_connection(A, 0.0)

    """
    if theta < 0:
        raise ValueError('expected a positive theta')
    if not sparse.issparse(A):
        raise ValueError('expected sparse matrix')
    if k < 0:
        raise ValueError('expected positive number of steps')
    if not isinstance(k, int):
        raise ValueError('expected integer')

    if A.format == 'bsr':
        bsr_flag = True
        numPDEs = A.blocksize[0]
        if A.blocksize[0] != A.blocksize[1]:
            raise ValueError('expected square blocks in BSR matrix A')
    else:
        bsr_flag = False
        numPDEs = 1

    # Convert A to csc and Atilde to csr
    if A.format == 'csr':
        Atilde = A.copy()
        A = A.tocsc()
    else:
        A = A.tocsc()
        Atilde = A.copy()
        Atilde = Atilde.tocsr()

    # Calculate the weighted-Jacobi parameter
    D = A.diagonal()
    Dinv = 1.0 / D
    Dinv[D == 0] = 0.0
    Dinv = sparse.csc_array((Dinv, (np.arange(A.shape[0], dtype=A.indptr.dtype),
                                    np.arange(A.shape[1], dtype=A.indptr.dtype))),
                            shape=A.shape)
    DinvA = Dinv @ A
    omega = 1.0 / approximate_spectral_radius(DinvA)
    del DinvA

    # Approximate A-inverse with k steps of w-Jacobi and a zero initial guess
    S = sparse.csc_array(A.shape, dtype=A.dtype)  # empty matrix
    Id = sparse.eye_array(A.shape[0], A.shape[1], format='csc')
    for _i in range(k + 1):
        S = S + omega * (Dinv @ (Id - A @ S))

    # Calculate the strength entries in S column-wise, but only strength
    # values at the sparsity pattern of A
    for i in range(Atilde.shape[0]):
        v = S[:, [i]].toarray()
        v = v.ravel()
        Av = A @ v
        denom = np.sqrt(np.inner(v.conj(), Av))
        # replace entries in row i with strength values
        for j in range(Atilde.indptr[i], Atilde.indptr[i + 1]):
            col = Atilde.indices[j]
            vj = v[col].copy()
            v[col] = 0.0
            #   =  (||v_j||_A - ||v||_A) / ||v||_A
            val = np.sqrt(np.inner(v.conj(), A @ v)) / denom - 1.0

            # Negative values generally imply a weak connection
            if val > -0.01:
                Atilde.data[j] = abs(val)
            else:
                Atilde.data[j] = 0.0

            v[col] = vj

    # Apply drop tolerance
    Atilde = classical_strength_of_connection(Atilde, theta=theta)
    Atilde.eliminate_zeros()

    # Put ones on the diagonal
    Atilde = Atilde + Id.tocsr()
    Atilde.sort_indices()

    # Amalgamate Atilde for the BSR case, using ones for all strong connections
    if bsr_flag:
        Atilde = Atilde.tobsr(blocksize=(numPDEs, numPDEs))
        nblocks = Atilde.indices.shape[0]
        uone = np.ones((nblocks,))
        Atilde = sparse.csr_array((uone, Atilde.indices, Atilde.indptr),
                                   shape=(int(Atilde.shape[0] / numPDEs),
                                          int(Atilde.shape[1] / numPDEs)))

    # Scale C by the largest magnitude entry in each row
    Atilde = scale_rows_by_largest_entry(Atilde)

    return Atilde


def ode_strength_of_connection(A, B=None, epsilon=4.0, k=2, proj_type='l2',
                               block_flag=False, symmetrize_measure=True):
    """Use evolution_strength_of_connection instead (deprecated)."""
    warn('ode_strength_of_connection method is deprecated. '
         'Use evolution_strength_of_connection.', DeprecationWarning, stacklevel=2)
    return evolution_strength_of_connection(A, B, epsilon, k, proj_type,
                                            block_flag, symmetrize_measure)


def evolution_strength_of_connection(A, B=None, epsilon=4.0, k=2,
                                     proj_type='l2', block_flag=False,
                                     symmetrize_measure=True):
    """Evolution strength measure.

    Construct strength of connection matrix using an Evolution-based measure.

    Parameters
    ----------
    A : csr_array, bsr_array
        Sparse NxN matrix.
    B : str, array
        If `B=None`, then the near nullspace vector used is all ones.  If `B` is
        an (NxK) array, then B is taken to be the near nullspace vectors.
    epsilon : scalar
        Drop tolerance.
    k : int
        ODE num time steps, step size is assumed to be `1/rho(DinvA)`.
    proj_type : {'l2','D_A'}
        Define norm for constrained min prob, i.e. define projection.
    block_flag : bool
        If True, use a block D inverse as preconditioner for A during
        weighted-Jacobi.
    symmetrize_measure : bool
        Symmetrize the strength as `(A + A.T) / 2`.

    Returns
    -------
    csr_array
        Sparse matrix of strength values.

    Notes
    -----
    See [1]_ for more details.

    References
    ----------
    .. [1] Olson, L. N., Schroder, J., Tuminaro, R. S.,
       "A New Perspective on Strength Measures in Algebraic Multigrid",
       submitted, June, 2008.

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.gallery import stencil_grid
    >>> from pyamg.strength import evolution_strength_of_connection
    >>> n=3
    >>> stencil =  np.array([[-1.0,-1.0,-1.0],
    ...                        [-1.0, 8.0,-1.0],
    ...                        [-1.0,-1.0,-1.0]])
    >>> A = stencil_grid(stencil, (n,n), format='csr')
    >>> S = evolution_strength_of_connection(A,  np.ones((A.shape[0],1)))

    """
    # ====================================================================
    # Check inputs
    if epsilon < 1.0:
        raise ValueError('expected epsilon > 1.0')
    if k <= 0:
        raise ValueError('number of time steps must be > 0')
    if proj_type not in ['l2', 'D_A']:
        raise ValueError('proj_type must be "l2" or "D_A"')
    if not sparse.issparse(A) or A.format not in ('csr', 'bsr'):
        raise TypeError('expected csr_array or bsr_array')

    # ====================================================================
    # Format A and B correctly.
    # B must be in mat format, this isn't a deep copy
    if B is None:
        Bmat = np.ones((A.shape[0], 1), dtype=A.dtype)
    else:
        Bmat = np.asarray(B)

    # Pre-process A.  We need A in CSR, to be devoid of explicit 0's and have
    # sorted indices
    if A.format != 'csr':
        csrflag = False
        numPDEs = A.blocksize[0]
        D = A.diagonal()
        # Calculate Dinv@A
        if block_flag:
            Dinv = get_block_diag(A, blocksize=numPDEs, inv_flag=True)
            Dinv = sparse.bsr_array((Dinv,
                                     np.arange(Dinv.shape[0], dtype=Dinv.indptr.dtype),
                                     np.arange(Dinv.shape[1], dtype=Dinv.indptr.dtype)),
                                    shape=A.shape)
            Dinv_A = (Dinv @ A).tocsr()
        else:
            Dinv = np.zeros_like(D)
            mask = D != 0.0
            Dinv[mask] = 1.0 / D[mask]
            Dinv[D == 0] = 1.0
            Dinv_A = scale_rows(A, Dinv, copy=True)
        A = A.tocsr()
    else:
        csrflag = True
        numPDEs = 1
        D = A.diagonal()
        Dinv = np.zeros_like(D)
        mask = D != 0.0
        Dinv[mask] = 1.0 / D[mask]
        Dinv[D == 0] = 1.0
        Dinv_A = scale_rows(A, Dinv, copy=True)

    A.eliminate_zeros()
    A.sort_indices()

    # Handle preliminaries for the algorithm
    dimen = A.shape[1]
    NullDim = Bmat.shape[1]

    # Get spectral radius of Dinv@A, this will be used to scale the time step
    # size for the ODE
    rho_DinvA = approximate_spectral_radius(Dinv_A)

    # Calculate D_A for later use in the minimization problem
    if proj_type == 'D_A':
        D_A = sparse.diags_array([D], offsets=[0], shape=(dimen, dimen), format='csr')
    else:
        D_A = sparse.eye_array(dimen, format='csr', dtype=A.dtype)

    # Calculate (I - delta_t Dinv A)^k
    #      In order to later access columns, we calculate the transpose in
    #      CSR format so that columns will be accessed efficiently
    # Calculate the number of time steps that can be done by squaring, and
    # the number of time steps that must be done incrementally
    nsquare = int(np.log2(k))
    ninc = k - 2**nsquare

    # Calculate one time step
    Id = sparse.eye_array(dimen, format='csr', dtype=A.dtype)
    Atilde = Id - (1.0 / rho_DinvA) * Dinv_A
    Atilde = Atilde.T.tocsr()

    # Construct a sparsity mask for Atilde that will restrict Atilde^T to the
    # nonzero pattern of A, with the added constraint that row i of Atilde^T
    # retains only the nonzeros that are also in the same PDE as i.
    mask = A.copy()

    # Restrict to same PDE
    if numPDEs > 1:
        row_length = np.diff(mask.indptr)
        my_pde = np.mod(np.arange(dimen), numPDEs)
        my_pde = np.repeat(my_pde, row_length)
        mask.data[np.mod(mask.indices, numPDEs) != my_pde] = 0.0
        del row_length, my_pde
        mask.eliminate_zeros()

    # If the total number of time steps is a power of two, then there is
    # a very efficient computational short-cut.  Otherwise, we support
    # other numbers of time steps, through an inefficient algorithm.
    if ninc > 0:
        warn('The most efficient time stepping for the Evolution Strength '
             f'Method is done in powers of two.\nYou have chosen {k} time steps.')

        # Calculate (Atilde^nsquare)^T = (Atilde^T)^nsquare
        for _i in range(nsquare):
            Atilde = Atilde @ Atilde

        JacobiStep = (Id - (1.0 / rho_DinvA) @ Dinv_A).T.tocsr()
        for _i in range(ninc):
            Atilde = Atilde @ JacobiStep
        del JacobiStep

        # Apply mask to Atilde, zeros in mask have already been eliminated at
        # start of routine.
        mask.data[:] = 1.0
        Atilde = Atilde.multiply(mask)
        Atilde.eliminate_zeros()
        Atilde.sort_indices()

    elif nsquare == 0:
        if numPDEs > 1:
            # Apply mask to Atilde, zeros in mask have already been eliminated
            # at start of routine.
            mask.data[:] = 1.0
            Atilde = Atilde.multiply(mask)
            Atilde.eliminate_zeros()
            Atilde.sort_indices()

    else:
        # Use computational short-cut for case (ninc == 0) and (nsquare > 0)
        # Calculate Atilde^k only at the sparsity pattern of mask.
        for _i in range(nsquare - 1):
            Atilde = Atilde @ Atilde

        # Call incomplete mat-mat mult
        AtildeCSC = Atilde.tocsc()
        AtildeCSC.sort_indices()
        mask.sort_indices()
        Atilde.sort_indices()
        amg_core.incomplete_mat_mult_csr(Atilde.indptr, Atilde.indices,
                                         Atilde.data, AtildeCSC.indptr,
                                         AtildeCSC.indices, AtildeCSC.data,
                                         mask.indptr, mask.indices, mask.data,
                                         dimen)

        del AtildeCSC, Atilde
        Atilde = mask
        Atilde.eliminate_zeros()
        Atilde.sort_indices()

    del Dinv, Dinv_A, mask

    # Calculate strength based on constrained min problem of
    # min( z - B@x ), such that
    # (B@x)|_i = z|_i, i.e. they are equal at point i
    # z = (I - (t/k) Dinv A)^k delta_i
    #
    # Strength is defined as the relative point-wise approx. error between
    # B@x and z.  We don't use the full z in this problem, only that part of
    # z that is in the sparsity pattern of A.
    #
    # Can use either the D-norm, and inner product, or l2-norm and inner-prod
    # to solve the constrained min problem.  Using D gives scale invariance.
    #
    # This is a quadratic minimization problem with a linear constraint, so
    # we can build a linear system and solve it to find the critical point,
    # i.e. minimum.
    #
    # We exploit a known shortcut for the case of NullDim = 1.  The shortcut is
    # mathematically equivalent to the longer constrained min. problem

    if NullDim == 1:
        # Use shortcut to solve constrained min problem if B is only a vector
        # Strength(i,j) = | 1 - (z(i)/b(j))/(z(j)/b(i)) |
        # These ratios can be calculated by diagonal row and column scalings

        # Create necessary vectors for scaling Atilde
        #   Its not clear what to do where B == 0.  This is an
        #   an easy programming solution, that may make sense.
        Bmat_forscaling = np.ravel(Bmat)
        Bmat_forscaling[Bmat_forscaling == 0] = 1.0
        DAtilde = Atilde.diagonal()
        DAtildeDivB = np.ravel(DAtilde) / Bmat_forscaling

        # Calculate best approximation, z_tilde, in span(B)
        #   Importantly, scale_rows and scale_columns leave zero entries
        #   in the matrix.  For previous implementations this was useful
        #   because we assume data and Atilde.data are the same length below
        data = Atilde.data.copy()
        Atilde.data[:] = 1.0
        Atilde = scale_rows(Atilde, DAtildeDivB)
        Atilde = scale_columns(Atilde, np.ravel(Bmat_forscaling))

        # If angle in the complex plane between z and z_tilde is
        # greater than 90 degrees, then weak.  We can just look at the
        # dot product to determine if angle is greater than 90 degrees.
        angle = np.multiply(np.real(Atilde.data), np.real(data)) +\
            np.multiply(np.imag(Atilde.data), np.imag(data))
        angle = angle < 0.0
        angle = np.array(angle, dtype=bool)

        # Calculate Approximation ratio
        Atilde.data = Atilde.data / data

        # If approximation ratio is less than tol, then weak connection
        weak_ratio = np.abs(Atilde.data) < 1e-4

        # Calculate Approximation error
        Atilde.data = abs(1.0 - Atilde.data)

        # Set small ratios and large angles to weak
        Atilde.data[weak_ratio] = 0.0
        Atilde.data[angle] = 0.0

        # Set near perfect connections to 1e-4
        Atilde.eliminate_zeros()
        Atilde.data[Atilde.data < np.sqrt(np.finfo(float).eps)] = 1e-4

        del data, weak_ratio, angle

    else:
        # For use in computing local B_i^H@B, precompute the element-wise
        # multiply of each column of B with each other column.  We also scale
        # by 2.0 to account for BDB's eventual use in a constrained
        # minimization problem
        BDBCols = int(np.sum(np.arange(NullDim + 1)))
        BDB = np.zeros((dimen, BDBCols), dtype=A.dtype)
        counter = 0
        for i in range(NullDim):
            for j in range(i, NullDim):
                BDB[:, counter] = 2.0 *\
                    (np.conjugate(np.ravel(Bmat[:, i])) * np.ravel(D_A @ Bmat[:, j]))
                counter = counter + 1

        # Choose tolerance for dropping "numerically zero" values later
        tol = set_tol(Atilde.dtype)

        # Use constrained min problem to define strength
        amg_core.evolution_strength_helper(Atilde.data,
                                           Atilde.indptr,
                                           Atilde.indices,
                                           Atilde.shape[0],
                                           np.ravel(Bmat),
                                           np.ravel((D_A @ B.conj()).T),
                                           np.ravel(BDB),
                                           BDBCols, NullDim, tol)

        Atilde.eliminate_zeros()

    # All of the strength values are real by this point, so ditch the complex
    # part
    Atilde.data = np.array(np.real(Atilde.data), dtype=float)

    # Apply drop tolerance
    if epsilon != np.inf:
        amg_core.apply_distance_filter(dimen, epsilon, Atilde.indptr,
                                       Atilde.indices, Atilde.data)
        Atilde.eliminate_zeros()

    # Symmetrize
    if symmetrize_measure:
        Atilde = 0.5 * (Atilde + Atilde.T)

    # Set diagonal to 1.0, as each point is strongly connected to itself.
    Id = sparse.eye_array(dimen, format='csr')
    Id.data -= Atilde.diagonal()
    Atilde = Atilde + Id

    # If converted BSR to CSR, convert back and return amalgamated matrix,
    #   i.e. the sparsity structure of the blocks of Atilde
    if not csrflag:
        Atilde = Atilde.tobsr(blocksize=(numPDEs, numPDEs))

        n_blocks = Atilde.indices.shape[0]
        blocksize = Atilde.blocksize[0] * Atilde.blocksize[1]
        CSRdata = np.zeros((n_blocks,))
        amg_core.min_blocks(n_blocks, blocksize,
                            np.ravel(np.asarray(Atilde.data)), CSRdata)
        # Atilde = sparse.csr_array((data, row, col), shape=(*,*))
        Atilde = sparse.csr_array((CSRdata, Atilde.indices, Atilde.indptr),
                                   shape=(int(Atilde.shape[0] / numPDEs),
                                          int(Atilde.shape[1] / numPDEs)))

    # Standardized strength values require small values be weak and large
    # values be strong.  So, we invert the algebraic distances computed here
    Atilde.data = 1.0 / Atilde.data

    # Scale C by the largest magnitude entry in each row
    Atilde = scale_rows_by_largest_entry(Atilde)

    return Atilde


def relaxation_vectors(A, R, k, alpha):
    """Generate test vectors by relaxing on Ax=0 for some random vectors x.

    Parameters
    ----------
    A : csr_array
        Sparse NxN matrix.
    R : int
        Number of random vectors.
    k : int
        Number of relaxation passes.
    alpha : scalar
        Weight for Jacobi.

    Returns
    -------
    array
        Dense array N x k array of relaxation vectors.

    """
    # random n x R block in column ordering
    n = A.shape[0]
    x = np.random.rand(n * R) - 0.5
    x = np.reshape(x, (n, R), order='F')
    # for i in range(R):
    #     x[:,i] = x[:,i] - np.mean(x[:,i])
    b = np.zeros((n, 1))

    for r in range(0, R):
        jacobi(A, x[:, r], b, iterations=k, omega=alpha)
        # x[:,r] = x[:,r]/norm(x[:,r])

    return x


def affinity_distance(A, alpha=0.5, R=5, k=20, epsilon=4.0):
    """Affinity Distance Strength Measure.

    Parameters
    ----------
    A : csr_array
        Sparse NxN matrix.
    alpha : scalar
        Weight for Jacobi.
    R : int
        Number of random vectors.
    k : int
        Number of relaxation passes.
    epsilon : scalar
        Drop tolerance.

    Returns
    -------
    csr_array
        Sparse matrix of strength values.

    Notes
    -----
    No unit testing yet.

    Does not handle BSR matrices yet.

    See [1]_ for more details.

    References
    ----------
    .. [1] Oren E. Livne and Achi Brandt, "Lean Algebraic Multigrid
        (LAMG): Fast Graph Laplacian Linear Solver"

    """
    if not sparse.issparse(A) or A.format != 'csr':
        A = sparse.csr_array(A)

    if alpha < 0:
        raise ValueError('expected alpha>0')

    if R <= 0 or not isinstance(R, int):
        raise ValueError('expected integer R>0')

    if k <= 0 or not isinstance(k, int):
        raise ValueError('expected integer k>0')

    if epsilon < 1:
        raise ValueError('expected epsilon>1.0')

    def distance(x):
        (rows, cols) = A.nonzero()
        return 1 - np.sum(x[rows] * x[cols], axis=1)**2 / \
            (np.sum(x[rows]**2, axis=1) * np.sum(x[cols]**2, axis=1))

    return distance_measure_common(A, distance, alpha, R, k, epsilon)


def algebraic_distance(A, alpha=0.5, R=5, k=20, epsilon=2.0, p=2):
    """Algebraic Distance Strength Measure.

    Parameters
    ----------
    A : csr_array
        Sparse NxN matrix.
    alpha : scalar
        Weight for Jacobi.
    R : int
        Number of random vectors.
    k : int
        Number of relaxation passes.
    epsilon : scalar
        Drop tolerance.
    p : scalar or inf
        The `p`-norm of the measure.

    Returns
    -------
    csr_array
        Sparse matrix of strength values.

    Notes
    -----
    No unit testing yet.

    Does not handle BSR matrices yet.

    See [1]_ for more details.

    References
    ----------
    .. [1] Ilya Safro, Peter Sanders, and Christian Schulz,
        "Advanced Coarsening Schemes for Graph Partitioning"

    """
    if not sparse.issparse(A) or A.format != 'csr':
        A = sparse.csr_array(A)

    if alpha < 0:
        raise ValueError('expected alpha>0')

    if R <= 0 or not isinstance(R, int):
        raise ValueError('expected integer R>0')

    if k <= 0 or not isinstance(k, int):
        raise ValueError('expected integer k>0')

    if epsilon < 1:
        raise ValueError('expected epsilon>1.0')

    if p < 1:
        raise ValueError('expected p>1 or equal to numpy.inf')

    def distance(x):
        (rows, cols) = A.nonzero()
        if p != np.inf:
            avg = np.sum(np.abs(x[rows] - x[cols])**p, axis=1) / R
            return (avg)**(1.0 / p)

        return np.abs(x[rows] - x[cols]).max(axis=1)

    return distance_measure_common(A, distance, alpha, R, k, epsilon)


def distance_measure_common(A, func, alpha, R, k, epsilon):
    """Strength of connection matrix from a function applied to relaxation vectors.

    Parameters
    ----------
    A : csr_array
        Input matrix for strength.
    func : callable
        Function to apply to relaxation vectors.
    alpha : scalar
        Weight for Jacobi.
    R : int
        Number of random vectors.
    k : int
        Number of relaxation passes.
    epsilon : scalar
        Filter tolerance.

    Returns
    -------
    array_like
        Test vectors.

    """
    # create test vectors
    x = relaxation_vectors(A, R, k, alpha)

    # apply distance measure function to vectors
    d = func(x)

    # drop distances to self
    (rows, cols) = A.nonzero()
    weak = np.where(rows == cols)[0]
    d[weak] = 0
    C = sparse.csr_array((d, (rows, cols)), shape=A.shape)
    C.eliminate_zeros()

    # remove weak connections
    # removes entry e from a row if e > theta * min of all entries in the row
    amg_core.apply_distance_filter(C.shape[0], epsilon, C.indptr,
                                   C.indices, C.data)
    C.eliminate_zeros()

    # Standardized strength values require small values be weak and large
    # values be strong.  So, we invert the distances.
    C.data = 1.0 / C.data

    # Put an identity on the diagonal
    C = C + sparse.eye_array(C.shape[0], C.shape[1], format='csr')

    # Scale C by the largest magnitude entry in each row
    C = scale_rows_by_largest_entry(C)

    return C
