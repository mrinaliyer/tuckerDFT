import sys
sys.path.insert(0,'../TensorLib')
import math
import numpy as np
import time
import FunctionalRayleighQuotientSeparable
import ProjectedHamiltonian
import scipy.optimize 
import scipy.linalg
import tensor
import DTA
import FEM
import sys, slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc


Print = PETSc.Sys.Print


#-----------------------------------------------------------------------------------
def meshgrid2(*arrs):
    ''' ij indexing returns same output as matlab '''
    arrs = tuple(reversed(arrs))  #edit
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

#    (indexing == 'ij'== matlab):
    return tuple(ans[::-1])

#------------------------------------------------------------------------------------------
def residual(guessFields,functional):
    ''' function to calculate force'''

    numberNodes = functional.fem.numberNodes
    nodalFields = np.zeros((numberNodes,3))
    # apply boundary conditions to trial solution (guessFields) to get nodalFields
    nodalFields[0,0:3] = 0
    nodalFields[1:-1,0] = guessFields[0:numberNodes-2]
    nodalFields[1:-1,1] = guessFields[numberNodes-2:2*(numberNodes - 2)]
    nodalFields[1:-1,2] = guessFields[2*(numberNodes-2):3*(numberNodes-2)]
    nodalFields[-1,0:3] = 0
    lagrangeMultiplier = float(guessFields[-1])

    F = functional.computeVectorizedForce(lagrangeMultiplier,nodalFields)

    return F

#----------------------------------------------------------------------------------------
def matrix(guessFields,functional):

    numberNodes = functional.fem.numberNodes
    nodalFields = np.zeros((numberNodes,3))
    # apply boundary conditions to trial solution (guessFields) to get nodalFields
    nodalFields[0,0:3] = 0
    nodalFields[1:-1,0] = guessFields[0:numberNodes-2]
    nodalFields[1:-1,1] = guessFields[numberNodes-2:2*(numberNodes - 2)]
    nodalFields[1:-1,2] = guessFields[2*(numberNodes-2):3*(numberNodes-2)]
    nodalFields[-1,0:3] = 0
    lagrangeMultiplier = float(guessFields[-1])

    [Hx,Hy,Hz,M] = functional.generateHamiltonianGenericPotential(nodalFields)

    return [Hx,Hy,Hz,M]

#---------------------------------------------------------------------------------
class SeparableHamiltonian :

    def __init__(self,functional):

        self.functional = functional
        numberNodes = functional.fem.numberNodes
        self.nodalFields = np.zeros((numberNodes,3))

    def residualPetsc(self,snes,X,F):

        nodalFields = self.nodalFields
        numberNodes = self.functional.fem.numberNodes

        # set boundary values
        nodalFields[0,0:3] = 0
        nodalFields[-1,0:3] = 0

        #
        # copy X into remainder of nodal fields
        # remember : X is identical to guessFields in usage
        #
        x = X.getArray()
        
        # set inner values
        for i in range(X.size):
            if i >= 0 and i < numberNodes - 2 :
                nodalFields[i+1,0] = x[i]

            elif i >= numberNodes -2 and i < 2*(numberNodes - 2):
                nodalFields[i-numberNodes+3,1] = x[i]

            elif i >= 2*(numberNodes-2) and i < 3*(numberNodes-2):
                nodalFields[i-2*numberNodes+5,2] = x[i]
    
            else:
                pass

            lagrangeMultiplier = x[X.size-1]


        # call forceEvaluator
        force = self.functional.computeVectorizedForce(lagrangeMultiplier,nodalFields)

        # copy force into F
        for index,val in enumerate(force):
            F.setValue(index,val)

        F.assemblyBegin()
        F.assemblyEnd()

        return

    def formInitGuess(self,snes,X,guessFields):


        for index,val in enumerate(guessFields):
            X.setValue(index,val)
        
        X.assemblyBegin()
        X.assemblyEnd()

        return

    def copySolution(self,X,output):
        lenX = X.size
        output = np.zeros((lenX))
        for i in range(lenX):
            output[i] = X[i]
        return output

#--------------------------------------------------------------------------------
def main():

       
    # 'linear;, 'quadratic' or 'cubic'
    elementType = 'cubic'

    # '2pt','3pt', '4pt'
    quadratureRule = '4pt'

    # meshType = 'uniform' or 'adaptive'
    meshType = 'uniform'

    # FIXME: generalize later
    numAtoms = 1

    if(numAtoms == 1):

      # define number of elements
      numberElements = 52

      # define domain size in 1D
      domainStart = -17.5
      domainEnd = 17.5

      #define parameters for adaptive Mesh
      innerDomainSize = 8
      innerMeshSize = 0.25

      #read the external potental from matlab
      if(meshType == 'uniform'):
        pot = open('vpotOneAtomUniform52Elem.txt','r')
      else :
        pot = open('vpotOneAtomAdaptive52Elem.txt','r')

      

    else:
      # define number of elements
      numberElements = 84

      # define domain size in 1D
      domainStart = -17.5
      domainEnd = 17.5

      #define parameters for adaptive Mesh
      innerDomainSize = 16
      innerMeshSize = 0.25

      
      #read the external potental from matlab
      if(meshType =='uniform'):
        pot = open('vpot14AtomUniform84Elem.txt','r')
      else :
	pot = open('vpot14AtomAdaptive84Elem.txt','r')

    # create FEM object
    fem=FEM.FEM(numberElements,
                quadratureRule,
                elementType,
                domainStart,
                domainEnd,
                innerDomainSize,
                innerMeshSize,
	        meshType)

      
    # mesh the domain in 1D
    globalNodalCoordinates = fem.generateNodes()
    
    # generate Adaptive Mesh
    numberNodes = fem.numberNodes

    
    #get number quadrature points
    numberQuadraturePoints = fem.getNumberQuadPointsPerElement()
    totalnumberQuadraturePoints = numberElements*numberQuadraturePoints
 
    v= []
    for line in pot:
      v.append(float(line))

    v = np.array(v)
    data= np.reshape(v,(totalnumberQuadraturePoints,totalnumberQuadraturePoints,totalnumberQuadraturePoints))


    # reducedRank
    rankVeff = 5
    reducedRank = (rankVeff, rankVeff, rankVeff)

    #perform tensor decomposition of effective potential
    time_tensorStart = time.clock()
    
     #convert to a tensor
    fTensor = tensor.tensor(data)
    [a,b] = DTA.DTA(fTensor,reducedRank)
    time_tensorEnd = time.clock()
    print 'time elapsed for tensor decomposition of Veff (s)',time_tensorEnd-time_tensorStart
    sigma_core = a.core
    sigma = sigma_core.tondarray()
    umat = a.u[0].real
    vmat = a.u[1].real
    wmat = a.u[2].real
    sigma = sigma.real
    sigma2D = sigma.reshape((rankVeff**2,rankVeff)) # done to reduce cost while contracting

    functional = FunctionalRayleighQuotientSeparable.FEMFunctional(fem,
                                                                   rankVeff,
                                                                   sigma2D,
                                                                   [umat,vmat,wmat])
       
       
    # initial guess for psix, psiy, psiz 
    xx = globalNodalCoordinates
    yy = xx.copy()
    zz = xx.copy()
    X, Y, Z = meshgrid2(xx,yy,zz)


    psixyz = (1/np.sqrt(np.pi))*np.exp(-np.sqrt(X**2 + Y**2 + Z**2))
    igTensor = tensor.tensor(psixyz)

    # generate tensor decomposition of above guess of rank 1
    time_tensorStart = time.clock()
    rankIG = (1,1,1)
    [Psi,Xi] = DTA.DTA(igTensor,rankIG)
    time_tensorEnd = time.clock()
    print 'time elapsed(s) for DTA of initial guess',time_tensorEnd-time_tensorStart


    guessPsix = Psi.u[0].real
    guessPsiy = Psi.u[1].real
    guessPsiz = Psi.u[2].real
    guessLM = 0.2
    guessFields = np.concatenate((guessPsix[1:-1],guessPsiy[1:-1],guessPsiz[1:-1]))

     
    #append the initial guess for lagrange multiplier
    guessFields = np.append(guessFields,guessLM)

    petsc = True

    if (petsc == False):
        
        t0 = time.clock()
        print 'Method fsolve'
        [result,infodict,ier,mesg] =  scipy.optimize.fsolve(residual,guessFields,args=(functional),full_output=1,xtol=1.0e-8,maxfev=20000)
        
        print 'Number of Function Evaluations: ',infodict['nfev']
        print 'Ground State Energy: ',result[-1]
        print 'numberElements ',numberElements
        print mesg
        t1 = time.clock()
        print 'time elapsed(s) for minimization problem',t1-t0
        

    else:
        t0 = time.clock()
        snes = PETSc.SNES().create() 
        pde = SeparableHamiltonian(functional)
        n = len(guessFields)
        F = PETSc.Vec().createSeq(n)
        snes.setFunction(pde.residualPetsc,F)
        snes.setUseMF()
        snes.getKSP().setType('gmres')
        snes.setFromOptions()
        X = PETSc.Vec().createSeq(n)
        pde.formInitGuess(snes, X,guessFields)
        snes.solve(None, X)
        print X[n-1]
        t1 = time.clock()
        print 'time elapsed(s) for minimization problem',t1-t0 
        result = pde.copySolution(X,guessFields)  

    #
    #define the rank for generating tucker basis for the wavefunctions
    #
<<<<<<< HEAD
    rankTuckerBasis =10
=======
    rankTuckerBasis = 20
>>>>>>> 341e1a5807de8d028cfeac33894323750bf11aa1

    #
    # separable psi_x, psi_y and psi_z are now stored in in the array result
    #
    [Hx,Hy,Hz,M] = matrix(result,functional)
    
    [w,eigenVx] = scipy.linalg.eigh(Hx,M,eigvals=(0,rankTuckerBasis))
    [w,eigenVy] = scipy.linalg.eigh(Hy,M,eigvals=(0,rankTuckerBasis))
    [w,eigenVz] = scipy.linalg.eigh(Hz,M,eigvals=(0,rankTuckerBasis))

    
    #
    #create the basis functions after solving eigenValue problems in each direction
    #
    boundaryValue = 0

    basisX = np.zeros((numberNodes,rankTuckerBasis))
    basisY = np.zeros((numberNodes,rankTuckerBasis))
    basisZ = np.zeros((numberNodes,rankTuckerBasis))

    basisX[0,0:rankTuckerBasis] = boundaryValue
    basisX[1:-1,0:rankTuckerBasis]=eigenVx[0:numberNodes-2,0:rankTuckerBasis]
    basisX[-1,0:rankTuckerBasis] = boundaryValue

    basisY[0,0:rankTuckerBasis] = boundaryValue
    basisY[1:-1,0:rankTuckerBasis]=eigenVy[0:numberNodes-2,0:rankTuckerBasis]
    basisY[-1,0:rankTuckerBasis] = boundaryValue

    basisZ[0,0:rankTuckerBasis] = boundaryValue
    basisZ[1:-1,0:rankTuckerBasis]=eigenVz[0:numberNodes-2,0:rankTuckerBasis]
    basisZ[-1,0:rankTuckerBasis] = boundaryValue

    #
    #compute the projected Hamiltonian from given the tucker basis functions
    #
    projectedHamiltonianTucker = ProjectedHamiltonian.ProjectedHamiltonian(fem,
                                                                           rankVeff,
                                                                           rankTuckerBasis,
                                                                           sigma,
                                                                           [umat,vmat,wmat],
                                                                           [basisX,basisY,basisZ])


    #
    #precompute the integrals involved in computing projected Hamiltonian
    #
    t0 = time.clock()
    MatX, MatY, MatZ, MatDX, MatDY, MatDZ = projectedHamiltonianTucker.computeOverlapKineticTuckerBasis()
    MatPotX, MatPotY, MatPotZ = projectedHamiltonianTucker.computeOverlapPotentialTuckerBasis()          
    t1 = time.clock()
    print 'time elapsed(s) for precomputation of integrals',t1-t0 

                 
    MatX  = np.ravel(MatX)
    MatY  = np.ravel(MatY)
    MatZ  = np.ravel(MatZ)
    MatDX = np.ravel(MatDX)
    MatDY = np.ravel(MatDY)
    MatDZ = np.ravel(MatDZ)
    MatPotX_unravel = np.zeros((rankVeff,rankTuckerBasis**2))
    MatPotY_unravel = MatPotX_unravel.copy()
    MatPotZ_unravel = MatPotX_unravel.copy()
    
    for iRankVeff in xrange(rankVeff):
        MatPotX_unravel[iRankVeff,:] = np.ravel(MatPotX[iRankVeff])
        MatPotY_unravel[iRankVeff,:] = np.ravel(MatPotY[iRankVeff])
        MatPotZ_unravel[iRankVeff,:] = np.ravel(MatPotZ[iRankVeff])
    
    matToTucker = np.zeros((rankTuckerBasis**3,3))
    count  = 0
    for xI in xrange(rankTuckerBasis):
        for yI in xrange(rankTuckerBasis):
            for zI in xrange(rankTuckerBasis):
                matToTucker[count,0] = xI
                matToTucker[count,1] = yI
                matToTucker[count,2] = zI
                count += 1

    numGlobalRows = rankTuckerBasis**3

    numColumns = rankTuckerBasis**3

    ProjH = PETSc.Mat().create(PETSc.COMM_WORLD)
    ProjH.setSizes([numGlobalRows,numGlobalRows])
    ProjH.setType('dense')
    ProjH.setUp()
    ProjH.setFromOptions()

    Istart, Iend = ProjH.getOwnershipRange()
    numLocalRows = Iend - Istart
    locProjH = np.zeros((numLocalRows,rankTuckerBasis**3))
    idxrows = np.array(range(Istart,Iend),dtype=PETSc.IntType)
    idxcols = np.array(range(0,rankTuckerBasis**3),dtype=PETSc.IntType)

  
    t0 = time.clock()

    xIndexList = np.zeros((rankTuckerBasis),'int64')
    yIndexList = xIndexList.copy()
    zIndexList = xIndexList.copy()

    for row in xrange(0,numLocalRows):

        xI = matToTucker[row+Istart,0]
        yI = matToTucker[row+Istart,1]
        zI = matToTucker[row+Istart,2]

        for iRank in xrange(rankTuckerBasis):
            xIndexList[iRank] = rankTuckerBasis*xI + iRank
            yIndexList[iRank] = rankTuckerBasis*yI + iRank
            zIndexList[iRank] = rankTuckerBasis*zI + iRank

        localMatX  = MatX[xIndexList]
        localMatY  = MatY[yIndexList]
        localMatZ  = MatZ[zIndexList]
        localMatDX = MatDX[xIndexList]
        localMatDY = MatDY[yIndexList]
        localMatDZ = MatDZ[zIndexList]

        dyad = np.outer(np.outer(localMatDX,localMatY),localMatZ)
        dyad += np.outer(np.outer(localMatX,localMatDY),localMatZ)
        dyad += np.outer(np.outer(localMatX,localMatY),localMatDZ)

        locProjH[row,:] = 0.5*(dyad).reshape((rankTuckerBasis**3))
        
        
        # potential
        dyadPotential = np.zeros(dyad.shape)
        for iRankVeff in xrange(rankVeff):
            localMatPotX = MatPotX_unravel[iRankVeff][xIndexList]
            for jRankVeff in xrange(rankVeff):
                localMatPotY = MatPotY_unravel[jRankVeff][yIndexList]
	        dyadTemp = np.outer(localMatPotX,localMatPotY)
                for kRankVeff in xrange(rankVeff):                
                    localMatPotZ = MatPotZ_unravel[kRankVeff][zIndexList]
                    dyadPotential += sigma[iRankVeff,jRankVeff,kRankVeff]*np.outer(dyadTemp,localMatPotZ)
        
        locProjH[row,:] +=  dyadPotential.reshape((rankTuckerBasis**3))
    t1 = time.clock()
    print 'time elapsed(s) for computing projected Hamiltonian',t1 - t0 

    t0 = time.clock()
    #for iRank in xrange(0,numLocalRows):
    #   for jRank in xrange(0,rankTuckerBasis**3):
    #     ProjH[iRank+Istart,jRank] = locProjH[iRank,jRank]
  
    
    ProjH.setValues(idxrows,idxcols,locProjH,0)
    t1 = time.clock()
    print 'time elapsed(s) for inserting values in Petsc Matrix ',t1 - t0 

    t0 = time.clock()
    ProjH.assemblyBegin()
    ProjH.assemblyEnd()     
    t1 = time.clock()
<<<<<<< HEAD
    print 'time elapsed(s) for computing projected Hamiltonian',t1-t0 
=======
    print 'time elapsed(s) for assembling Projected Hamiltonian ',t1 - t0

   
>>>>>>> 341e1a5807de8d028cfeac33894323750bf11aa1
    xr, tmp = ProjH.getVecs()
    xi, tmp = ProjH.getVecs()
    nev = 4
    w = np.zeros(nev)    

    E = SLEPc.EPS(); E.create()

    E.setOperators(ProjH)
    E.setDimensions(nev,PETSc.DECIDE)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    E.setType(SLEPc.EPS.Type.JD)
    E.setFromOptions()

    E.solve()

    its = E.getIterationNumber()
    Print("Number of iterations of the method: %i" % its)
    sol_type = E.getType()
    Print("Solution method: %s" % sol_type)
    nconv = E.getConverged()
    Print("Number of converged eigenpairs: %d" % nconv)

    if nconv > 0:
        Print("")
        Print("        k          ||Ax-kx||/||kx|| ")
        Print("----------------- ------------------")
        for i in range(nev):
            k = E.getEigenpair(i, xr, xi)
            w[i] = k.real
            error = E.computeRelativeError(i)
            if k.imag != 0.0:
              Print(" %9f%+9f j  %12g" % (k.real, k.imag, error))
            else:
              Print(" %12f       %12g" % (k.real, error))
        Print("")


    #[w,eig] = scipy.linalg.eigh(locProjH,eigvals=(0,4))
    #print w

    occ = [1,1.0/6.0, 1.0/6.0, 1.0/6.0]
    bandenergy = 0;
    for k in range(4):
        bandenergy = bandenergy + 2*occ[k]*w[k];
    
    print 'bandenergy ',bandenergy
    print 'rankTuckerBasis' , rankTuckerBasis
   
  
    return





#
#
#
if __name__=="__main__":
    main()

