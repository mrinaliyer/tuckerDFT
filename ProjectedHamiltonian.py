import FEM
import math
import scipy as np
import sys

  #
  # Fem Functional : most imp function in this code
  #

class ProjectedHamiltonian:
        
    def __init__(self,
                 fem,
                 rankVeff,
                 rankTuckerBasis,
                 sigma,
                 tuckerDecomposedVeff,
                 bases):

        
        self.fem = fem
                                 
        self.rankVeff = rankVeff

        self.rankTuckerBasis = rankTuckerBasis

        self.sigma = sigma

        self.umat = tuckerDecomposedVeff[0]
        
        self.vmat = tuckerDecomposedVeff[1]

        self.wmat = tuckerDecomposedVeff[2]

        

        #compute quadPointValues for the basis Functions
        (self.basisXQuadValues,self.basisDXQuadValues) = fem.computeFieldsAtAllQuadPoints(bases[0])
        (self.basisYQuadValues,self.basisDYQuadValues) = fem.computeFieldsAtAllQuadPoints(bases[1])
        (self.basisZQuadValues,self.basisDZQuadValues) = fem.computeFieldsAtAllQuadPoints(bases[2])

          
    # 
    # 
    #
    def computeOverlapKineticTuckerBasis(self):
                                  

        fem = self.fem
        rankTuckerBasis = self.rankTuckerBasis

        #
        # do some initializations
        #
        MatX = np.zeros((rankTuckerBasis,rankTuckerBasis))
        MatY = MatX.copy()
        MatZ = MatY.copy()
        MatDX = MatX.copy()
        MatDY = MatY.copy()
        MatDZ = MatZ.copy()


       #
       #evaluate the requisite integrals
       #
        for I in range(0,rankTuckerBasis):
           fieldxI = self.basisXQuadValues[I,:]
           fieldDxI = self.basisDXQuadValues[I,:]
           fieldyI = self.basisYQuadValues[I,:]
           fieldDyI = self.basisDYQuadValues[I,:]
           fieldzI = self.basisZQuadValues[I,:]
           fieldDzI = self.basisDZQuadValues[I,:]
           for J in range(0,rankTuckerBasis):
              if(I <= J):
                  fieldxJ = self.basisXQuadValues[J,:]
                  fieldDxJ = self.basisDXQuadValues[J,:]
                  fieldyJ = self.basisYQuadValues[J,:]
                  fieldDyJ = self.basisDYQuadValues[J,:]
                  fieldzJ = self.basisZQuadValues[J,:]
                  fieldDzJ = self.basisDZQuadValues[J,:]
                  MatX[I,J] = fem.getIntegral1D(fieldxI*fieldxJ)
                  MatDX[I,J] = fem.getInvIntegral1D(fieldDxI*fieldDxJ)
                  MatY[I,J] = fem.getIntegral1D(fieldyI*fieldyJ)
                  MatDY[I,J] = fem.getInvIntegral1D(fieldDyI*fieldDyJ)
                  MatZ[I,J] =  fem.getIntegral1D(fieldzI*fieldzJ)
                  MatDZ[I,J] = fem.getInvIntegral1D(fieldDzI*fieldDzJ)
              else:
                  MatX[I,J] = MatX[J,I]
                  MatY[I,J] = MatY[J,I]
                  MatZ[I,J] = MatZ[J,I]
                  MatDX[I,J] = MatDX[J,I]
                  MatDY[I,J] = MatDY[J,I]
                  MatDZ[I,J] = MatDZ[J,I]

            
        return (MatX,MatY,MatZ,MatDX,MatDY,MatDZ)

    # 
    # 
    #
    def computeOverlapPotentialTuckerBasis(self):

        fem = self.fem
        rankTuckerBasis = self.rankTuckerBasis
        rankVeff = self.rankVeff

        #
        # do some initializations
        #
        MatPotX = np.zeros((rankVeff,rankTuckerBasis,rankTuckerBasis))
        MatPotY = MatPotX.copy()
        MatPotZ = MatPotY.copy()

        #
        #evaluate the requisite integrals
        #
        for irank in range(0,rankVeff):
            potentialTuckerx = self.umat[:,irank]
            potentialTuckery = self.vmat[:,irank]
            potentialTuckerz = self.wmat[:,irank]
            for I in range(0,rankTuckerBasis):
                fieldxI = self.basisXQuadValues[I,:]
                fieldyI = self.basisYQuadValues[I,:]
                fieldzI = self.basisZQuadValues[I,:]
                for J in range(0,rankTuckerBasis):
                   if(I <= J):
                       fieldxJ = self.basisXQuadValues[J,:]
                       fieldyJ = self.basisYQuadValues[J,:]
                       fieldzJ = self.basisZQuadValues[J,:]
                       MatPotX[irank,I,J] = fem.getIntegral1D(potentialTuckerx*fieldxI*fieldxJ)
                       MatPotY[irank,I,J] = fem.getIntegral1D(potentialTuckery*fieldyI*fieldyJ)
                       MatPotZ[irank,I,J]= fem.getIntegral1D(potentialTuckerz*fieldzI*fieldzJ)
                   else:
                       MatPotX[irank,I,J] = MatPotX[irank,J,I]
                       MatPotY[irank,I,J] = MatPotY[irank,J,I]
                       MatPotZ[irank,I,J] = MatPotZ[irank,J,I]
                       
        return (MatPotX,MatPotY,MatPotZ)

        
    

            
        
