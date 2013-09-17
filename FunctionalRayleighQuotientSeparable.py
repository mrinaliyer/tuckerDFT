import FEM
import math
import scipy as np
import sys

  #
  # Fem Functional : most imp function in this code
  #

class FEMFunctional:
        
    def __init__(self,
                 fem,
                 rankVeff,
                 sigma,
                 tuckerDecomposedVeff):
        
        self.fem = fem
                                 
        self.rankVeff = rankVeff

        self.sigma = sigma

        self.umat = tuckerDecomposedVeff[0]
        
        self.vmat = tuckerDecomposedVeff[1]

        self.wmat = tuckerDecomposedVeff[2]
        
        self.vShapeFunctionAtQuadPoints = np.tile(fem.shapeFunctionAtQuadPoints,(1,fem.numberElements))
        self.vShapeFunctionDerivsAtQuadPoints = np.tile(fem.shapeFunctionDerivsAtQuadPoints,(1,fem.numberElements))

        # extra data structs used while computing force
        numberNodes = fem.numberNodes
        numberNodesPerElement = fem.getNumberNodesPerElement()
        numberQuadraturePoints = fem.getNumberQuadPointsPerElement()
        numberElements = fem.numberElements

        self.Fx = np.zeros((numberNodes))
        self.Fy = np.zeros((numberNodes))
        self.Fz = np.zeros((numberNodes))

        self.X0 = np.zeros((numberNodesPerElement,numberQuadraturePoints*numberElements))

        self.Y0 = np.zeros((numberNodesPerElement,numberQuadraturePoints*numberElements))
        
        self.Z0 = np.zeros((numberNodesPerElement,numberQuadraturePoints*numberElements))

    #
    # static condensation
    #
    def condenseMatrix(self,H):
        
        # applyBoundaryConditions on Hx Hy Hz
        H = np.delete(H,0,0)
        H = np.delete(H,-1,0)
        H = np.delete(H,0,1)
        H = np.delete(H,-1,1)

        return H
        
    # 
    # 
    #
    def computeIntegralPsiSquare(self,
                                 psiQuadValues,
                                 DPsiQuadValues):

        fem = self.fem

        #
        #evaluate normPsix
        #
        quadValuesx = psiQuadValues[0,:]

        normPsix = fem.getIntegral1D(quadValuesx*quadValuesx)

        #
        #evaluate normPsiy
        #
        quadValuesy = psiQuadValues[1,:]
        normPsiy = fem.getIntegral1D(quadValuesy*quadValuesy)

        #
        #evaluate normPsiz
        #
        quadValuesz= psiQuadValues[2,:]
        normPsiz = fem.getIntegral1D(quadValuesz*quadValuesz)


        #
        #evaluate normDPsix
        #
        quadValuesDpsix = DPsiQuadValues[0,:]
        normDPsix = fem.getInvIntegral1D(quadValuesDpsix*quadValuesDpsix)

        #
        #evaluate normDPsiy
        #
        quadValuesDpsiy = DPsiQuadValues[1,:]
        normDPsiy = fem.getInvIntegral1D(quadValuesDpsiy*quadValuesDpsiy)
  
        
        #
        #evaluate normDPsiz
        #
        quadValuesDpsiz = DPsiQuadValues[2,:]
        normDPsiz = fem.getInvIntegral1D(quadValuesDpsiz*quadValuesDpsiz)
       
        
        return (normPsix,normPsiy,normPsiz,normDPsix,normDPsiy,normDPsiz)



        
    def computeSeparablePotentialsUsingTuckerVeff(self,
                                                  psiQuadValues):
                                                   
        
        fem = self.fem
        numberQuadPoints = fem.getNumberQuadPointsPerElement()
        numberElements = fem.numberElements
        rankVeff = self.rankVeff
        umat = self.umat
        vmat = self.vmat
        wmat = self.wmat
        sigma = self.sigma

        #
        #allocate storage for some integrals
        #
        intUiPsixPsix = np.zeros((rankVeff))
        intVjPsiyPsiy = np.zeros((rankVeff))
        intWkPsizPsiz = np.zeros((rankVeff))
                                                
        #
        #first precompute integrals of u_i(x)*psix*psix where i = 1 to R
        #R denotes rank used in tucker decomposition of Veff
        #
        a = psiQuadValues[0,:]
        b = psiQuadValues[1,:]
        c = psiQuadValues[2,:]

        aa = a*a
        bb = b*b
        cc = c*c

        for i in range(0,rankVeff):
            
            quadValuesx = aa*umat[:,i]

            intUiPsixPsix[i] = fem.getIntegral1D(quadValuesx)

        #
        #second precompute integrals of v_j(x)*psiy*psiy where j = 1 to R
        #R denotes rank used in tucker decomposition of Veff
        #    
        for j in range(0,rankVeff):
            
            quadValuesy = bb*vmat[:,j]
            
            intVjPsiyPsiy[j] = fem.getIntegral1D(quadValuesy)
            
        #
        #second precompute integrals of w_k(x)*psiz*psiz where k = 1 to R
        #R denotes rank used in tucker decomposition of Veff
        #    
        for k in range(0,rankVeff):
            
            quadValuesz = cc*wmat[:,k]
                                     
            intWkPsizPsiz[k] = fem.getIntegral1D(quadValuesz)
            

        vx = np.zeros((numberElements*numberQuadPoints))
        vy = np.zeros((numberElements*numberQuadPoints))
        vz = np.zeros((numberElements*numberQuadPoints))

        # e.g for a quadratic potential v = x^2+y^2+z^2
        #vx = 0.5*x*x*integral(psiy*psiy)*integral(psiz*psiz) +
        #       0.5*integral(y*y*psiy*psiy)*integral(psiz*psiz) +
        #       0.5*integral(psiy*psiy)*integral(z*z*psiz*psiz)
        #vy and vz similarly follows
                             

        for index in range(0,numberElements*numberQuadPoints):
       

            dyadX = np.outer(np.outer(umat[index,:],intVjPsiyPsiy),intWkPsizPsiz)
            vx[index]+= np.sum(sigma*dyadX)


            dyadY = np.outer(np.outer(vmat[index,:],intUiPsixPsix),intWkPsizPsiz)
            vy[index] += np.sum(sigma*dyadY)

            dyadZ = np.outer(np.outer(wmat[index,:],intUiPsixPsix),intVjPsiyPsiy)
            vz[index] += np.sum(sigma*dyadZ)
            

 

        return(vx,vy,vz)


    def computeVectorizedForce(self,lagrangeMultiplier,nodalFields):


        
        (psiQuadValues,DPsiQuadValues) = self.fem.computeFieldsAtAllQuadPoints(nodalFields)

        norms = self.computeIntegralPsiSquare(psiQuadValues,DPsiQuadValues)

        (vx,vy,vz) = self.computeSeparablePotentialsUsingTuckerVeff(psiQuadValues)

        # data members
        fem                              = self.fem
        numberNodes                      = fem.getNumberNodes()
        numberNodesPerElement            = fem.getNumberNodesPerElement()
        numberQuadraturePoints           = fem.getNumberQuadPointsPerElement()
        numberElements                   = fem.numberElements
        weightQuadPointValues            = fem.weightQuadPointValues
        jacobianQuadPointValues          = fem.jacobianQuadPointValues
        invJacobianQuadPointValues       = fem.invJacobianQuadPointValues
        vShapeFunctionAtQuadPoints       = self.vShapeFunctionAtQuadPoints
        vShapeFunctionDerivsAtQuadPoints = self.vShapeFunctionDerivsAtQuadPoints
        elementConnectivity              = fem.elementConnectivity
        
        Fx = self.Fx; Fy = self.Fy; Fz = self.Fz
        X0 = self.X0; Y0 = self.Y0; Z0 = self.Z0
        
        # Fx Fy Fz = 1*numNodes    
        Fx.fill(0)
        Fy.fill(0)
        Fz.fill(0)

        X0.fill(0)
        Y0.fill(0)
        Z0.fill(0)
        
        normPsix = norms[0]; normPsiy = norms[1]; normPsiz = norms[2];
        normDPsix = norms[3]; normDPsiy = norms[4]; normDPsiz = norms[5];

        cx = 1.0/(normPsiy*normPsiz)
        cy = 1.0/(normPsiz*normPsix)
        cz = 1.0/(normPsix*normPsiy)
        Tx = 0.5*(normDPsiy/normPsiy + normDPsiz/normPsiz)
        Ty = 0.5*(normDPsiz/normPsiz + normDPsix/normPsix)
        Tz = 0.5*(normDPsix/normPsix + normDPsiy/normPsiy)  

        for iNode in xrange(0,numberNodesPerElement):

             kinetic = 0.5*DPsiQuadValues[0,:]
             other   = cx*psiQuadValues[0,:]*vx+ (lagrangeMultiplier+Tx)*psiQuadValues[0,:]
 
             X0[iNode,:] = weightQuadPointValues*(kinetic*invJacobianQuadPointValues*vShapeFunctionDerivsAtQuadPoints[iNode,:] + jacobianQuadPointValues*vShapeFunctionAtQuadPoints[iNode,:]*other)

             kinetic = 0.5*DPsiQuadValues[1,:]
             other   = cy*psiQuadValues[1,:]*vy+ (lagrangeMultiplier+Ty)*psiQuadValues[1,:]
             
             Y0[iNode,:] = weightQuadPointValues*(kinetic*invJacobianQuadPointValues*vShapeFunctionDerivsAtQuadPoints[iNode,:] + jacobianQuadPointValues*vShapeFunctionAtQuadPoints[iNode,:]*other)

             kinetic = 0.5*DPsiQuadValues[2,:]
             other   = cz*psiQuadValues[2,:]*vz+ (lagrangeMultiplier+Tz)*psiQuadValues[2,:]
             
             Z0[iNode,:] = weightQuadPointValues*(kinetic*invJacobianQuadPointValues*vShapeFunctionDerivsAtQuadPoints[iNode,:] + jacobianQuadPointValues*vShapeFunctionAtQuadPoints[iNode,:]*other)
             

        for e in range(0,numberElements):          
            start        = e*numberQuadraturePoints
            end          = (e+1)*numberQuadraturePoints
            for iNode in xrange(0,numberNodesPerElement):
                iGlobal      = elementConnectivity[iNode,e]
                Fx[iGlobal] += np.sum(X0[iNode,start:end])
                Fy[iGlobal] += np.sum(Y0[iNode,start:end])
                Fz[iGlobal] += np.sum(Z0[iNode,start:end])

                
        F = np.concatenate((Fx[1:-1],Fy[1:-1],Fz[1:-1]))

        # append constraint force
        constraintForce = normPsix*normPsiy*normPsiz - 1.0
        F = np.append(F,constraintForce)
        

        return F

    def generateHamiltonianGenericPotential(self,
                                            nodalFields):
        
        
        (psiQuadValues,DPsiQuadValues) = self.fem.computeFieldsAtAllQuadPoints(nodalFields)

        norms = self.computeIntegralPsiSquare(psiQuadValues,DPsiQuadValues)
            
        (vx,vy,vz) = self.computeSeparablePotentialsUsingTuckerVeff(psiQuadValues)

        normx = norms[0]; normy = norms[1]; normz = norms[2];

        # data members
        fem = self.fem
        numberNodes = fem.getNumberNodes()
        numberNodesPerElement = fem.getNumberNodesPerElement()
        numberQuadraturePoints = fem.getNumberQuadPointsPerElement()
        numberElements = fem.numberElements
        weightQuadPointValues = fem.weightQuadPointValues
        jacobianQuadPointValues = fem.jacobianQuadPointValues
        invJacobianQuadPointValues = fem.invJacobianQuadPointValues
        shapeFunctionAtQuadPoints = fem.shapeFunctionAtQuadPoints
        shapeFunctionDerivsAtQuadPoints = fem.shapeFunctionDerivsAtQuadPoints
        elementConnectivity = fem.elementConnectivity

        Hx = np.zeros((numberNodes,numberNodes));
        Hy = Hx.copy();
        Hz = Hy.copy();
        M =  Hx.copy();

        elementStiffnessMatX    = np.zeros((numberNodesPerElement,numberNodesPerElement));
        elementStiffnessMatY    = elementStiffnessMatX.copy()
        elementStiffnessMatZ    = elementStiffnessMatX.copy()
        elementMassMat          = elementStiffnessMatX.copy()
        # 
        for e in range(numberElements):

            start = e*numberQuadraturePoints
            end   = (e+1)*numberQuadraturePoints
            
            for iNode in range(numberNodesPerElement):
                
                shapeFunctionGradientQuadPointValuesI = shapeFunctionDerivsAtQuadPoints[iNode,:]
                shapeFunctionQuadPointValuesI         = shapeFunctionAtQuadPoints[iNode,:]

                for jNode in range(numberNodesPerElement):

                    shapeFunctionGradientQuadPointValuesJ = shapeFunctionDerivsAtQuadPoints[jNode,:]
                    shapeFunctionQuadPointValuesJ         = shapeFunctionAtQuadPoints[jNode,:]


                    quadraturePointValuesKinetic = (1.0/2.0)*shapeFunctionGradientQuadPointValuesI*shapeFunctionGradientQuadPointValuesJ

                    
                    quadraturePointValuesPotX = (1/(normy*normz))*vx[start:end]*shapeFunctionQuadPointValuesI*shapeFunctionQuadPointValuesJ


                    quadraturePointValuesPotY = (1/(normx*normz))*vy[start:end]*shapeFunctionQuadPointValuesI*shapeFunctionQuadPointValuesJ
                    
                    quadraturePointValuesPotZ = (1/(normx*normy))*vz[start:end]*shapeFunctionQuadPointValuesI*shapeFunctionQuadPointValuesJ
                    
                    quadraturePointValuesMass = shapeFunctionQuadPointValuesI*shapeFunctionQuadPointValuesJ;

                    elementalInvJacAtQuadPoints = invJacobianQuadPointValues[start:end]

                    elementalJacAtQuadPoints = jacobianQuadPointValues[start:end]


                    shapeFunctionGradientIntegralIJ = fem.integrate(quadraturePointValuesKinetic,
                                                                    elementalInvJacAtQuadPoints);
         
                    elementStiffnessMatX[iNode,jNode] = shapeFunctionGradientIntegralIJ +fem.integrate(quadraturePointValuesPotX,elementalJacAtQuadPoints)

                    elementStiffnessMatY[iNode,jNode] = shapeFunctionGradientIntegralIJ \
                    +fem.integrate(quadraturePointValuesPotY,elementalJacAtQuadPoints)

                    elementStiffnessMatZ[iNode,jNode] = shapeFunctionGradientIntegralIJ \
                    +fem.integrate(quadraturePointValuesPotZ,elementalJacAtQuadPoints)

                    elementMassMat[iNode,jNode] = fem.integrate(quadraturePointValuesMass,elementalJacAtQuadPoints);


            #
            # insert into global stiffness matrix
            #

            for iNode in range(numberNodesPerElement):
                iGlobal = elementConnectivity[iNode,e]

                for jNode in range(numberNodesPerElement):

                    jGlobal = elementConnectivity[jNode,e];

                    Hx[iGlobal,jGlobal] +=  elementStiffnessMatX[iNode,jNode]
                    Hy[iGlobal,jGlobal] +=  elementStiffnessMatY[iNode,jNode]
                    Hz[iGlobal,jGlobal] +=  elementStiffnessMatZ[iNode,jNode]
                    M[iGlobal,jGlobal]  +=  elementMassMat[iNode,jNode];
        
        Hx = self.condenseMatrix(Hx)
        Hy = self.condenseMatrix(Hy)
        Hz = self.condenseMatrix(Hz)
        M = self.condenseMatrix(M)

        return (Hx,Hy,Hz,M)
