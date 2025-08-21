# ------------------------------------------------------------------
# Step 1: import scipy and pyamg packages
# ------------------------------------------------------------------
import numpy as np
import pyamg
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from scipy import sparse
from pyamg import strength

def show_levels(ml, name = None, lenres = None):

  # notice that there are 5 (or maybe 6) levels in the hierarchy
  #
  # we can look at the data in each of the levels
  # e.g. the multigrid components on the finest (0) level
  #      A: operator on level 0
  #      P: prolongation operator mapping from level 1 to level 0
  #      R: restriction operator mapping from level 0 to level 1
  #      B: near null-space modes for level 0
  #      presmoother: presmoothing function taking arguments (A,x,b)
  #      postsmoother: postsmoothing function taking arguments (A,x,b)

  print("\n")
  print("The Multigrid Hierarchy")
  if name != None:
    print(name)
  cycle_complexity = ml.cycle_complexity()
  print('cycle complexity: ' + str(cycle_complexity))
  if lenres!= None:
    print('iterations to converge: ' +str(lenres) )
    cost = lenres * cycle_complexity
    print('total cost: ' + str(cost))

  print("-----------------------")
  for l in range(len(ml.levels)):
    An = ml.levels[l].A.shape[0]
    Am = ml.levels[l].A.shape[1]

    if l == (len(ml.levels)-1):
      print(f"A_{l}: {An:>10}x{Am:<10}")
    else:
      Pn = ml.levels[l].P.shape[0]
      Pm = ml.levels[l].P.shape[1]
      print(f"A_{l}: {An:>10}x{Am:<10}   P_{l}: {Pn:>10}x{Pm:<10}")

def plot(plt, figname='out'):
  print('\07')
  import sys
  if '--savefig' in sys.argv:
    plt.savefig(figname, bbox_inches='tight', dpi=150)
  else:
    plt.show()

def plot_cost(ml,residuals,names):

  fig, ax = plt.subplots()
  for i in range(len(ml)):
    
    cycle_complexity = ml[i].cycle_complexity()
    res = residuals[i]
    iterations = len(res)
    
    ax.scatter(i,iterations*cycle_complexity, label=names[i])

  plt.legend()
  plot(plt)



def plot_C(ml, title=None):

  fig, ax = plt.subplots(2)
  for i in range(len(ml)):
    c = ml[i].levels[0].C
    ax[0].plot( c.data ,linestyle='-.', label=names[i])
    ax[1].plot( c.indices,linestyle='-.', label=names[i])

  ax[0].set_ylabel('Connection data')
  ax[1].set_ylabel('Connection indices')
  plt.legend()
  if title != None:
    plt.title(title)
  plot(plt)

def plot_P(ml, title=None):

  fig, ax = plt.subplots( 2 )
  if title != None:
    plt.title(title)
  for i in range(len(ml)):
    p = ml[i].levels[0].P.tocsr()
    ax[0].plot( p.data ,linestyle='-.', label=names[i])
    ax[1].plot( p.indices,linestyle='-.', label=names[i])

  ax[0].set_ylabel('Prolongation data')

  ax[1].set_ylabel('Prolongation indices')
  #ax.grid(True)
  #plt.legend()

  plot(plt)

def plot_AggOp(ml, title=None):

  fig, ax = plt.subplots(2)
  if title != None:
    fig.title(title)
  for i in range(len(ml)):
    p = ml[i].levels[0].AggOp.tocsr()
    #ax[0].plot( p.data ,linestyle='-.', label=names[i])
    ax[1].plot( p.indices,linestyle='-.', label=names[i])
    ax[0].scatter(ml[i].levels[0].AggOp.indices, ml[i].levels[0].AggOp.data,linestyle='-.', label=names[i])

  #ax.set_xlabel('indices')
  
  ax[0].set_ylabel('AggOp data')
  ax[1].set_ylabel('AggOp indices')
  #ax.set_ylabel('data')
  #ax.grid(True)
  plt.legend()

  plot(plt)


def plot_Cnodes(ml, title=None):

  fig, ax = plt.subplots()
  for i in range(len(ml)):
    
    ax.plot(ml[i].levels[0].Cnodes,linestyle='-.', label=names[i])

  ax.set_xlabel('Index')
  ax.set_ylabel('Cnodes')
  ax.grid(True)
  plt.legend()
  if title != None:
    plt.title(title)

  plot(plt)


def plot_residuals(res, title = None):

  fig, ax = plt.subplots()
  for i in range(len(ml)):
    ax.semilogy(res[i],linestyle='-.', label=names[i])

  ax.set_xlabel('Iteration')
  ax.set_ylabel('Relative Residual')
  ax.grid(True)
  plt.legend()
  if title != None:
    plt.title(title)

  plot(plt)

def test_diffusion(hier,n, eps,theta,epsmax = 0,epsinc = 1, thetamax= 0, thetainc=1):
  solver = hier[0]
  argvec = hier[1]
  
  solves =[ [] for i in range(len(argvec))]
  T = [t for t in np.arange(theta,thetamax,thetainc)]
  E = [ e for e in np.arange(eps,epsmax,epsinc)]
  x = []
  y = []
  z = []
  zzz = dict()
  for t in T:
    for e in E:
      stencil = pyamg.gallery.diffusion_stencil_2d(type='FE', epsilon=e, theta=np.pi /t)
      A = pyamg.gallery.stencil_grid(stencil, (n, n), format='csr')
      ml = []
      names = []
      keys = list(zzz)

      for kwargs in argvec:
        ml.append( solver( A, **kwargs ) )
        if str(kwargs) not in keys:
          zzz[str(kwargs)] = {'iter':[],'complexity':[],'cost':[]}
        names.append( str( kwargs ) )


      res = [[]for i in range(len(ml))]
      [ml[i].solve(b, tol=tolerance, residuals=res[i]) for i in range(len(ml))]
      x.append(t)
      y.append(e)
      z.append(len(res[-1]))
      [zzz[names[i]]['iter'].append(len(res[i])) for i in range(len(names))]
      [zzz[names[i]]['complexity'].append(ml[i].cycle_complexity()) for i in range(len(names))]
      [zzz[names[i]]['cost'].append(zzz[names[i]]['iter'][-1]*zzz[names[i]]['complexity'][-1]) for i in range(len(names))]
      title = names[-1]

      solves.append( (names, ml, res, "Diffusion eps={} theta=pi/{}".format(e,t)) )
  for name in list(zzz):
    zlabel = 'cost'
    z = zzz[name][zlabel]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x,y,z)
    #surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    #                       linewidth=0, antialiased=False)
    ax.set_xlabel('diffusion theta')
    ax.set_ylabel('diffusion epsilon')
    ax.set_zlabel(zlabel)

    plt.title(name)
    plt.show()

  
  return solves

# ------------------------------------------------------------------
# Step 2: setup up the system using pyamg.gallery
# ------------------------------------------------------------------
n = 20
X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))

#set problem
stencil = pyamg.gallery.diffusion_stencil_2d(type='FE', epsilon=0.05, theta=np.pi / 5)
A = pyamg.gallery.stencil_grid(stencil, (n, n), format='csr')
#fig, ax = plt.subplots()
#ax.plot(A.data,linestyle='--',label="A.data")
#plot(plt)
#A = pyamg.gallery.poisson((n,n),format='csr')
B = None
#A, B = pyamg.gallery.linear_elasticity((n, n), format='csr')

#plot_pairwise_soc(A)


#A = data['A'].tocsr()

# set right hand side
b = np.random.rand(A.shape[0])                     # pick a random right hand side
#b = np.ones(A.shape[0])

#set solver
air = pyamg.air_solver
ruge_stuben = pyamg.ruge_stuben_solver
smoothed_aggregation = pyamg.smoothed_aggregation_solver
solvers = [smoothed_aggregation, ruge_stuben, air]
solverid = 0
solver = solvers[solverid]

#set tolerance
tolerance = 1e-10
#set threshold value
theta =1- 0.001
thetamin = theta
thetainc = 1
thetamax = 1

replacezerosmin = 1
replacezerosmax = 2
replacezerosinc = 1

''' 
damping factor for jacobi/richardson
scaled by spectral radius at each level, so must be in range (0,2)
'''

omegainc = 0.5
omegamax = 2
omega = omegainc


presmoother = []
smoothers = [('jacobi',{'omega':omega}) for omega in np.arange(omega,omegamax,omegainc)]
smoothers.extend( [('richardson',{'omega':omega}) for omega in np.arange(omega,omegamax,omegainc)] )
#smoothers.append('energy')

#ps = [('pairwise',{'theta':theta,'replacezeros':i}) for i in range(4)]

strength = [('classical',{'theta':theta}),('pairwise',{'theta':theta,'replacezeros':3}),'symmetric','evolution','energy_based','algebraic_distance']
cs = [('classical',{'theta':theta}) for theta in np.arange(thetamin,thetamax,thetainc)]
strength = cs
#strength = []



for theta in np.arange(thetamin,thetamax,thetainc):
  for replace in range(replacezerosmin, replacezerosmax):
    strength.append( ('pairwise',{'theta':theta,'replacezeros':replace,'smooth':0}) )
    #strength.append( ('pairwise',{'theta':theta,'replacezeros':0,'smooth':1},'classical',{'theta':theta}) )
    strength.append( ('pairwise',{'theta':theta,'replacezeros':0,'smooth':1}))
    strength.append(('pairwise',{'theta':theta,'replacezeros':0,'smooth':0}) )
    #strength.append( ('classical',{'theta':theta}, 'pairwise',{'theta':theta,'replacezeros':0,'smooth':0}) )

aggregates = ['standard','naive',('pairwise',{'matchings':1})]

cfsplittings = ['RS' for i in range(6)]

# ------------------------------------------------------------------
# Step 3: setup of the multigrid hierarchy
# ------------------------------------------------------------------
names = []
ml = []
argvec = []
if solverid == 0:
  for s in strength:
    for agg in aggregates[:2]:
      for smooth in smoothers[:1]: 
        kwargs = {'strength':s,'aggregate':agg, 'keep':1}
        argvec.append(kwargs)
        '''kwargs = {'strength':s,'aggregate':agg,
                  'presmoother':smooth,
                  'postsmoother':smooth,'keep':1}
        '''
        names.append( str( kwargs ) )
        #print(kwargs)
        #ml.append( solver( A,B, **kwargs ) )
elif solverid == 1:
  for s in strength:
    kwargs = {'strength':s, 'keep':1}
    names.append( str( kwargs ) )
    print(kwargs)
    ml.append( solver( A, **kwargs ) )

else:
  ml1 = solver(A,strength=('classical',{'theta':theta}),keep=1)   # construct the multigrid hierarchy
  ml2 = solver(A,strength=('pairwise',{'theta':theta}),keep=1)   # construct the multigrid hierarchy
  ml3 = solver(A,strength=('classical',{'theta':theta}),interpolation='direct',keep=1)   # construct the multigrid hierarchy
  ml4 = solver(A,strength=('pairwise',{'theta':theta}), CF=('RS',{'second_pass':1}) , keep = 1)# construct the multigrid hierarchy
  ml5 = solver(A,strength=('pairwise',{'theta':theta, 'replacezeros':'classical'}), keep=1 )   # construct the multigrid hierarchy
  ml6 = solver(A,strength=('pairwise',{'theta':theta,'replacezeros':'classical'}),interpolation='direct', keep=1 )   # construct the multigrid hierarchy


  ml = [ml1,ml2,ml3,ml4,ml5,ml6]
hier = (solver, argvec)
# ------------------------------------------------------------------
# Step 4: solve the system
# ------------------------------------------------------------------
#res = [[]for i in range(len(ml))]
#x = [ml[i].solve(b, tol=tolerance, residuals=res[i]) for i in range(len(ml))]
diffusion =1
if diffusion:
  X = test_diffusion(hier, n, 0,np.pi/9, 0.1,0.001, np.pi,np.pi/9)
else:
  X = [(names,ml,res,"title")]
# ------------------------------------------------------------------
# Step 5: print details
# ------------------------------------------------------------------
for x in X:
  names = x[0]
  ml = x[1]
  res = x[2]
  title = x[3]
  print(title)
  [print( len( ml[i].levels ) ) for i in range(len(ml))]

  [print( show_levels( ml[i], names[i], len(res[i]) )) for i in range(len(ml))]

  # ------------------------------------------------------------------
  # Step 8: plot
  # ------------------------------------------------------------------
  plot_cost(ml,res,names)
  plot_residuals(res, title=title)
  plot_C(ml, title=title)
  plot_P(ml, title=title)

  if solver  ==0:

    plot_Cnodes(ml, title=title)
    plot_AggOp(ml, title = title)
# ------------------------------------------------------------------
# Step 6: change the hierarchy
# ------------------------------------------------------------------


# we can also change the details of the hierarchy
'''
ml = pyamg.smoothed_aggregation_solver(A,  # the matrix
                                       B=X.reshape(n * n, 1),             # the representation of the near null space (this is a poor choice)
                                       BH=None,                           # the representation of the left near null space
                                       symmetry='symmetric',              # indicate that the matrix is Hermitian
                                       strength='pairwise',              # change the strength of connection
                                       aggregate='standard',              # use a standard aggregation method
                                       smooth=('jacobi', {'omega': 4.0 / 3.0, 'degree': 2}),   # prolongation smoothing
                                       presmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
                                       postsmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
                                       improve_candidates=[('block_gauss_seidel',
                                                           {'sweep': 'symmetric', 'iterations': 4}), None],
                                       max_levels=10,                     # maximum number of levels
                                       max_coarse=5,                      # maximum number on a coarse level
                                       keep=False)                        # keep extra operators around in the hierarchy (memory)
'''

# ------------------------------------------------------------------
# Step 8: plot
# ------------------------------------------------------------------
plot_residuals(res)
plot_C(ml)
plot_P(ml)

if solver  ==0:

  plot_Cnodes(ml)
  plot_AggOp(ml)

