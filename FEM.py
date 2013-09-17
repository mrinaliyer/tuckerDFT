import math
import scipy as np
import sys

class FEM :
    # class defining FEM operations, specialized to 1D
    

    def __init__(self,
                 numberElements,
                 quadratureRule,
                 elementType,
                 domainStart,
                 domainEnd,
                 innerDomainSize,
                 innerMeshSize,
		 meshType):
                 
        self.numberElements = numberElements
        self.quadratureRule = quadratureRule
        self.elementType    = elementType
        self.domainStart    = domainStart
        self.domainEnd      = domainEnd
        self.innerDomainSize = innerDomainSize
        self.innerMeshSize = innerMeshSize
        self.numberNodes    = self.getNumberNodes()
        self.shapeFunctionAtQuadPoints = self.generateShapeFunctions()
        self.shapeFunctionDerivsAtQuadPoints = self.generateShapeFunctionGradient()
        self.elementConnectivity = self.generateElementConnectivity()
        self.innerDomainSize = innerDomainSize
	self.innerMeshSize   = innerMeshSize
        self.meshType        = meshType

        (self.jacobianQuadPointValues,self.invJacobianQuadPointValues, \
         self.positionQuadPointValues,self.weightQuadPointValues) = \
         self.generateQuadPointData(self.shapeFunctionAtQuadPoints,
                                    self.shapeFunctionDerivsAtQuadPoints)


    #
    # calculate number of nodes in domain using number of elements and element type
    #
    def  getNumberNodes(self):
        
        if self.elementType == "linear":
            numberNodes = self.numberElements*1 +1
        elif self.elementType == "quadratic":
            numberNodes = self.numberElements*2 + 1
        elif self.elementType == "cubic":
            numberNodes = self.numberElements*3 +1
        else:
            print "elementType "+elementType+" not implemented"
            
        return numberNodes
   
    #
    # call either adaptive or uniform mesh
    #
    def generateNodes(self):
	if(self.meshType =='adaptive'):
	  return self.generateNodesAdaptive()
	else :
	  return self.generateNodesUniform()

    #
    # seed nodes uniformly within domainStart and domainEnd. 
    # This function is "naive" and will work only for a restricted case of uniform mesh and
    # conventional FEM elements (i.e no spectral). For adaptive refinement, we need to go element
    # by element and insert nodes in them
    #
    def generateNodesUniform(self):        
        numberNodes = self.getNumberNodes()
        nodes = np.linspace(self.domainStart,self.domainEnd,numberNodes)
        return nodes

    def generateNodesAdaptive(self):
        innerDomainSize = self.innerDomainSize
        innerMeshSize   = self.innerMeshSize
        numberElementsInnerDomain = innerDomainSize/innerMeshSize
	assert(numberElementsInnerDomain < self.numberElements)
        domainCenter = (self.domainStart+self.domainEnd)/2
        nodes0 = np.linspace(domainCenter,innerDomainSize/2.0,(numberElementsInnerDomain/2.0)+1.0)
        nodes0 = np.delete(nodes0,-1)
        numberOuterIntervalsFromDomainCenter = (self.numberElements - numberElementsInnerDomain)/2.0
        const = np.log2(innerDomainSize/2.0)/0.5
        exp = np.linspace(const,np.log2(self.domainEnd*self.domainEnd),numberOuterIntervalsFromDomainCenter+1)
        nodes1 = np.power(np.sqrt(2),exp)
        nodesp = np.concatenate((nodes0,nodes1))
        nodesn = -nodesp[::-1]
        nodesn = np.delete(nodesn,-1)
        linNodalCoordinates = np.concatenate((nodesn,nodesp))
        nodalCoordinates = 0

        #Introduce higher order nodes
        if self.elementType == "quadratic" or self.elementType == "cubic":
           if self.elementType == "quadratic":
              numberNodesPerElement = 3 
           elif self.elementType == "cubic":
              numberNodesPerElement = 4

           for i in range(0,len(linNodalCoordinates)-1):
              newnodes = np.linspace(linNodalCoordinates[i],linNodalCoordinates[i+1],numberNodesPerElement)
              nodalCoordinates = np.delete(nodalCoordinates,-1)
              nodalCoordinates = np.concatenate((nodalCoordinates,newnodes))

        else:
           nodalCoordinates = linNodalCoordinates
    
        return nodalCoordinates


    #
    # generate element connectivity
    # node numbering = [0,1,2,3,4,5 ....n]
    #
    def generateElementConnectivity(self):
        
        numberNodesPerElement = self.getNumberNodesPerElement() 
        # create as an int array numberNodesPerElement*numberElements
        elementConnectivity = np.zeros((numberNodesPerElement,self.numberElements),'int64')
        
        for i in range(0,numberNodesPerElement):
            for j in range(0, self.numberElements):
                elementConnectivity[i,j]=j*(numberNodesPerElement-1)+i

        return elementConnectivity    

    #
    # get number nodes per element
    #
    def getNumberNodesPerElement(self):
        
        if self.elementType == "linear":
            numberNodesPerElement = 2
        elif self.elementType == "quadratic":
            numberNodesPerElement = 3
        elif self.elementType == "cubic":
            numberNodesPerElement = 4
        else :
            print "elementType " + self.elementType + " not implemented"
            
        return numberNodesPerElement

    #
    # get number quadraturePoints per element
    #
    def getNumberQuadPointsPerElement(self):
        
        if self.quadratureRule == "2pt":
            numQuadPoints = 2
        elif self.quadratureRule == "3pt":
            numQuadPoints = 3
        elif self.quadratureRule == "4pt":
            numQuadPoints = 4
        else:
            print "quadrature rule " + self.quadratureRule + " is not implemented"
    
        return numQuadPoints

    #
    # location of quad points in parent element
    #
    def getQuadPointCoordinates(self):
        
        if self.quadratureRule == "2pt":
            quadPtCoords = np.array([-1.0/math.sqrt(3),1.0/math.sqrt(3)])
        elif self.quadratureRule == "3pt":
            quadPtCoords = np.array([-math.sqrt(15.0)/5.0 , 0.0 , math.sqrt(15.0)/5.0])
        elif self.quadratureRule == "4pt":
            quadPtCoords = np.array([-0.861136311594052575223946488893,-0.339981043584856264802665759103,0.339981043584856264802665759103,0.861136311594052575223946488893])
        else:
            print "quadrature rule " + self.quadratureRule + " is not implemented"
    
        return quadPtCoords

    #
    # get parent nodal coordinates
    #
    def getBiUnitNodalCoordinates(self):
        
        if self.elementType == "linear":
            coords = np.array([-1.0 , 1.0 ])
        elif self.elementType == "quadratic":
            coords = np.array([-1.0 ,0.0 ,1.0 ])
        elif self.elementType == "cubic":
            coords = np.array([-1.0, -1.0/3.0, 1.0/3.0, 1.0])
    
        return coords

    #
    # quadrature weights
    #
    def getQuadratureWeights(self):
        
        if self.quadratureRule == "2pt":
            weights = np.array([1,1])
        elif self.quadratureRule == "3pt":
            weights = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
        elif self.quadratureRule == "4pt":
            weights = np.array([0.347854845137453857373063949222,0.652145154862546142626936050778, 0.652145154862546142626936050778,0.347854845137453857373063949222])
        else:
            print "quadrature rule " + self.quadratureRule + " is not implemented"
    
        return weights

    #
    # integrator
    #
    def integrate(self,quadraturePointValues, jacobianValues):
        
        weights = self.getQuadratureWeights()
        value = np.sum(weights*quadraturePointValues*jacobianValues)

        return value


    #
    # evaluate vectorized integral over the complete domain (i.e. the previous
    # function integrates within an element, but this function evaluates the 
    # interal over the whole domain)
    #
    def getIntegral1D(self,
                      quadPointValues):

     
        integralVector = self.weightQuadPointValues*quadPointValues*self.jacobianQuadPointValues
        value = np.sum(integralVector)

        return value

    #
    # for some cases, e.e with gradients, the integral required is qpc*weights*invJac
    # 
    def getInvIntegral1D(self,
                         quadPointValues):

 
        integralVector = self.weightQuadPointValues*quadPointValues*self.invJacobianQuadPointValues
        value = np.sum(integralVector)

        return value



    
    #
    #  get shape function gradients at quad points of parent element
    #
    def generateShapeFunctionGradient(self):
        
        quadPointCoordinates = self.getQuadPointCoordinates()
        
        nodalCoordinates = self.getBiUnitNodalCoordinates()
        
        numberNodesPerElement = self.getNumberNodesPerElement()
        
        numberQuadPoints = self.getNumberQuadPointsPerElement()

        shapeFunctionGradient = np.zeros((numberNodesPerElement,numberQuadPoints))
        for i in range(0,numberNodesPerElement):
            # generate shape function for node i
            xi = nodalCoordinates[i]
            shapeFunction = [1]
            for j in range(0,numberNodesPerElement):
                if (j != i):
                    xj = nodalCoordinates[j]
                    shapeFunction = self.poly_multiply(shapeFunction,[-xj/(xi-xj) , 1.0/(xi-xj)])
            # differentiate this to get derivative
            shapeFunctionDerivative = self.poly_derivative(shapeFunction)
            # evaluate this function at all quad pts to get shape function grad
            for k in range(0,numberQuadPoints):
                shapeFunctionGradient[i,k] = self.poly_eval(shapeFunctionDerivative,
                                                       quadPointCoordinates[k])

    
        return shapeFunctionGradient


    #
    # get shape functions at quad points of parent element
    #
    def generateShapeFunctions(self):

        quadPointCoordinates = self.getQuadPointCoordinates()

        nodalCoordinates = self.getBiUnitNodalCoordinates()
    
        numberNodesPerElement = self.getNumberNodesPerElement()

        numberQuadPoints = self.getNumberQuadPointsPerElement()

        shapeFunctionMatrix = np.zeros( (numberNodesPerElement,
                                         numberQuadPoints))
    
        for i in range(0,numberNodesPerElement):
        #generate shape function for node i
            xi = nodalCoordinates[i]
            shapeFunction = [1]
            for j in range(0,numberNodesPerElement):
                if (j != i):
                    xj = nodalCoordinates[j]
                    shapeFunction = self.poly_multiply(shapeFunction,[-xj/(xi-xj), 1.0/(xi-xj)])
        #evaluate this shape function at all quad points and fill up matrix
            for k in range(0,numberQuadPoints):
                shapeFunctionMatrix[i,k] = self.poly_eval(shapeFunction,
                                                          quadPointCoordinates[k])

        return shapeFunctionMatrix

    #
    #interpolate derivatives of shape functions 
    #
    def interpolateDerivatives(self,
                               nodalValues):

        interpolatedDerivatives = np.dot(nodalValues,self.shapeFunctionDerivsAtQuadPoints)

        return interpolatedDerivatives


    #
    # same as above. use internal copy of shape functions
    #
    def interpolateFunction(self,
                            nodalValues):
        interpolatedValues = np.dot(nodalValues,self.shapeFunctionAtQuadPoints)
        
        return interpolatedValues


    #
    #compute Jacobian at QuadPoints
    #
    def computeJacobianAtQuadPoints(self,
                                    nodalCoordinates):

        jacobian = self.interpolateDerivatives(nodalCoordinates)

        return jacobian

    #
    # for an element, fine location of its nodes
    #
    def getLocalNodalCoordinates(self,
                                 elementId,
                                 globalNodalCoordinates,
                                 elementConnectivity) :
        
        localNodeIds = elementConnectivity[:,elementId]
        localNodeCoordinates = globalNodalCoordinates[localNodeIds]

        return localNodeCoordinates


    #
    # poly eval
    #
    def poly_eval(self,
                  plist,
                  x):
        """ Evaluate plist(polynomial in list form) at x"""
        value = 0.0
        for i in range(len(plist)):
            value= value + plist[i]*math.pow(x,i)

        return value

    #
    # poly derivative
    #
    def poly_derivative(self,plist):
        
        derivative = []
        if not plist: return derivative
        for i in range(1,len(plist)):
            derivative.append(i*plist[i])

        return derivative

    def add(self,p1,p2):
        "Return a new poly-list corresponding to the sum of the two input lists"
        if len(p1)> len(p2):
            new =[i for i in p1]
            for i in range(len(p2)): new[i] += p2[i]
        else:
            new = [i for i in p2]
            for i in range(len(p1)): new[i] += p1[i]
        return new

    
    #
    # poly multiply
    #
    def poly_multiply(self,p1,p2):

        if len(p1) > len(p2) : short,long = p2,p1
        else : short,long = p1,p2

        new = []
        for i in range(len(short)):
            temp = [0]*i # increment the list with i zeros
            for t in long :
                temp.append(t*short[i])
        
            new = self.add(new,temp)

        return new

    #
    # generate QuadPoint Data for all elements
    #
    #
    def generateQuadPointData(self,
                              shapeFunctionAtQuadPoints,
                              shapeFunctionDerivsAtQuadPoints):

        weights = self.getQuadratureWeights()
        globalNodalCoordinates = self.generateNodes()
        numberQuadraturePoints = len(weights)
        elementConnectivity = self.generateElementConnectivity()
        jacobianQuadPointValues = np.zeros((self.numberElements*numberQuadraturePoints))
        invJacobianQuadPointValues = np.zeros((self.numberElements*numberQuadraturePoints))
        positionQuadPointValues = np.zeros((self.numberElements*numberQuadraturePoints))
        weightQuadPointValues = np.zeros((self.numberElements*numberQuadraturePoints))

        for e in range(0,self.numberElements):
            elementId = e
            #get nodal coordinates in elementId
            localNodalCoordinates = self.getLocalNodalCoordinates(elementId,
                                                                  globalNodalCoordinates,
                                                                  elementConnectivity)
            #compute Jacobian at Quadrature Points
            elemJacobQuadPointValues = self.computeJacobianAtQuadPoints(localNodalCoordinates)

            #compute spatial-Coordinates of Quadrature Points
            elemPositionQuadPointValues = self.interpolateFunction(localNodalCoordinates)


            #Fill up data
            start = e*numberQuadraturePoints
            end   = (e+1)*numberQuadraturePoints

            jacobianQuadPointValues[start:end]  = elemJacobQuadPointValues

            invJacobianQuadPointValues[start:end] = 1./elemJacobQuadPointValues

            positionQuadPointValues[start:end] = elemPositionQuadPointValues

            weightQuadPointValues[start:end] = weights

        return (jacobianQuadPointValues, invJacobianQuadPointValues, positionQuadPointValues,weightQuadPointValues)

    #
    # interpolate field (given in nodalFields) to all the quad points in the domain and return as
    # an unrolled loop
    #
    def computeFieldsAtAllQuadPoints(self,
                                     nodalFields):

        shapeFunctionDerivsAtQuadPoints = self.shapeFunctionDerivsAtQuadPoints
        shapeFunctionAtQuadPoints       = self.shapeFunctionAtQuadPoints
        elementConnectivity             = self.elementConnectivity
        numberQuadraturePoints          = self.getNumberQuadPointsPerElement()
        numberNodesPerElement           = self.getNumberNodesPerElement()
        numberElements                  = self.numberElements
        numberNodes                     = self.numberNodes
        weightQuadPointValues           = self.weightQuadPointValues
        jacobianQuadPointValues         = self.jacobianQuadPointValues
        invJacobianQuadPointValues      = self.invJacobianQuadPointValues
  

        shape = nodalFields.shape
        numberComponents = shape[1]
        numberTuples = shape[0]
        assert(numberTuples == numberNodes)

        #
        # compute fields at quad points
        #
        fQuadValues = np.zeros((numberComponents,numberElements*numberQuadraturePoints))
        DfQuadValues = np.zeros((numberComponents,numberElements*numberQuadraturePoints))

        #
        #get localNode Ids
        #
        for e in range(0,numberElements):
        
            localNodeIds = elementConnectivity[:,e]
            start = e*numberQuadraturePoints
            end   = (e+1)*numberQuadraturePoints
         
            for iComponent in range(numberComponents):
                localNodalValues = (nodalFields[:,iComponent])[localNodeIds]
                fQuadValues[iComponent,start:end]=self.interpolateFunction(localNodalValues)
                DfQuadValues[iComponent,start:end]=self.interpolateDerivatives(localNodalValues)
     
        return (fQuadValues,DfQuadValues)
    
    
#
#
if __name__=="__main__":
    main()


