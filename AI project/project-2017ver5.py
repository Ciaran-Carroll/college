## -------------------------------------------------------------------------
##
## CE4708 - Artificial Intelligence
## Semester 1 - Autumn 2017/18
##
## 23-11-2017
##
## Group Members:-
## Paul Lynch (16123778)
## Ciaran Carroll (13113259)
## Kevin Burke (14155893)
## 
## Develop a program in Python to implement a soft-margin, kernel-based
## Support Vector Machine (SVM) classifier
##
## Outline:-
## This project implements a radial basis function kernel with 
## sigma^2 = 0.25.
## The program is trained using the training-dataset-aut-2017.txt file 
## and subsequently tested using testing-dataset-aut-2017.txt.
##
## Two values are used for the C parameter, C = 1.0 and C = 10^6
## The Python CVXopt library is used to implement the QP-solver function
##
##


from cvxopt import matrix, solvers, sparse
from math import exp
import numpy as np
import matplotlib.pyplot as plt


## Load training and testing datasets from txt files
## Split files into input (Xs2) and expected output (Ts2)
## Break up input (Xs2) into X and Y input values.
dataset = np.loadtxt('training-dataset-aut-2017.txt')
##dataset = np.loadtxt('testing-dataset-aut-2017.txt')
Xs2, Ts2 = dataset[:, :2], dataset[:, 2]
X = dataset[:, :1]
Y = dataset[:, 1:2]

##  Define C parameter for classifier.
##  Test the dataset with C = 1.0 and C = 10^6
C = 1.0
##C = 1000000.0


##--------------------------------------------------------------------------
##
##  rbfKernel
##
##  Return the radial basis function kernel exp(-||x-y||^2/2*sigma^2).  Note
##  sigma2 = 0.25, as specified in the project outline
##
##  Typical problems not too sensitive to size of variance, but note, it
##  must be +ve.
##

def rbfKernel(v1, v2, sigma2=0.25):
    assert len(v1) == len(v2)
    assert sigma2 >= 0.0
    mag2 = sum(map(lambda x, y: (x - y) * (x - y), v1, v2))  ## Squared mag of diff.
    return exp(-mag2 / (2.0 * sigma2))


## Make the P matrix for a nonlinear, kernel-based SVM problem.
##
def makeP(xs, ts, K):
    """Make the P matrix given the list of training vectors,
       desired outputs and kernel."""
    N = len(xs)
    assert N == len(ts)
    P = matrix(0.0, (N, N), tc='d')
    for i in range(N):
        for j in range(N):
            P[i, j] = ts[i] * ts[j] * K(xs[i], xs[j])
    return P


##--------------------------------------------------------------------------
##
##  makeLambdas
##
##  Use the qp solver from the cvx package to find a list of Lagrange
##  multipliers (lambdas or L's) for an Xs/Ts problem, where Xs is a list
##  of input vectors (themselves represented as simple lists) and Ts a list
##  of desired outputs.
##
##
##  Note that we are trying to solve the problem:
##
##      Maximize:
##
##        W(L) =  \sum_i L_i
##                - 1/2 sum_i sum_j L_i * L_j * t_i * t_j * K(x_i,x_j)
##
##      subject to:  \sum_i t_i L_i  =  0   and   L_i >= 0.
##
##
##  but the "standard" quadratic programming problem is subtly different,
##  it attempts to *minimize* the following quadratic form:
##
##        f(y) = 1/2 y^t P y  +  q^t y
##
##  subject to:  G L <= h   and   A y = b, where P is an n x n
##  symmetric matrix, G is a 2n x n matrix, A is a 1 x n
##  (row) vector, q is an n x 1 (column) vector, as are h and y.
##  N.B., Vector y is the solution being searched for.
##
##  To turn the W(l) constrained maximazation into a constrained
##  minimization of f, it suffices to set:
##
##          [-1.0]
##             .
##             .
##      q = [-1.0]      (n element column vector).
##          [-1.0]
##             .
##             .
##          [-1.0]
##
##
##          [-1.0,  0.0  ....  0.0]
##          [ 0.0, -1.0           ]
##          [             .       ]   (2n x n matrix with -1.0 on
##      G = [             .       ]    diagonal 1, +1.0 on diagonal 3
##          [            0.0,     ]    and 0.0's everywhere else. 
##          [           -1.0,  0.0]    )
##          [            0.0, -1.0]
##          [+1.0,  0.0  ....  0.0]
##          [ 0.0, +1.0        0.0]
##          [        .            ]
##          [        .            ]
##          [           +1.0      ]
##          [                 +1.0]
##
##
##
##      A = [ t_1, t_2, t_3, ... t_n],  a row vector with n elements
##                                      made using the t list input.
##
##
##      h = 2n element column vector of 0.0's and C's.
##
##      b = [0.0], i.e., a 1 x 1 matrix containing 0.0.
##
##
##          [                    ]   (n x n matrix with elements
##      P = [ t_i t_j K(x_i x_j) ]    t_i t_j x_i x_j).
##          [                    ]
##
##
##  The solution (if one exists) is returned by the "qp" solver as
##  a vector of elements.  The solver actually returns a dictionary
##  of values, this contains lots of information about the solution,
##  quality, etc.  But, from the point of view of this routine the
##  important part is the vector of "l" values, which is accessed
##  under the 'x' key in the returned dictionary.
##
##  N.B.  All the routines in the Cxvopt library are very sensitive to
##  the data types of their arguments.  In particular, all vectors,
##  matrices, etc., passed to "qp" must have elements that are
##  doubles.

def makeLambdas(Xs, Ts, K=rbfKernel):
    "Solve constrained maximization problem and return list of l's."
    P = makeP(Xs, Ts, K)            ## Build the P matrix.
    n = len(Ts)                     ## Length of Dataset, used to create other matrices
    q = matrix(-1.0, (n, 1))        ## This builds an n-element column
                                    ## vector of -1.0's (note the double-
                                    ## precision constant)
    h1 = matrix(0.0, (n, 1))        ## create one column matrix of n 0's

    h2 = matrix(C, (n, 1))          ## create one column matrix of n C's

    hcon = np.concatenate([h1, h2])
    h = matrix(hcon)                ## Concatenate h1 and h2 matrices to
	                                ## to create h matrix

    G1 = matrix(0.0, (n, n))        ## These lines generate G1, an

    G1[::(n + 1)] = -1.0            ## n x n matrix with -1.0's on its
                                    ## main diagonal

    G2 = matrix(0.0, (n, n))        ## These lines generate G, an
    G2[::(n + 1)] = 1.0             ## n x n matrix with 1.0's on its
                                    ## main diagonal.

    Gcon = np.concatenate((G1, G2)) ## This line concatenates G1 and G2
                                    ## creating a 2n x n matrix with
                                    ## -1.0s on diagonal 1 and 1.0s on
                                    ## diagonal 2.
    G = matrix(Gcon)

    A = matrix(Ts, (1, n), tc='d')
    ##
    ## Now call "qp". Details of the parameters to the call can be
    ## found in the online cvxopt documentation.
    ##
    r = solvers.qp(P, q, G, h, A, matrix(0.0))  ## "qp" returns a dict, r.
    ##
    ## Return results. Return a tuple, (Status,Ls).  First element is
    ## a string, which will be "optimal" if a solution has been found.
    ## The second element is a list of Lagrange multipliers for the problem,
    ## rounded to six decimal digits to remove algorithm noise.
    ##
    Ls = [round(l, 6) for l in list(r['x'])]  ## "L's" are under the 'x' key.
    return (r['status'], Ls)

##--------------------------------------------------------------------------
##
##  Find the bias for this kernel-based classifier.
##
##  The bias can be generated from any support vector Xs[n]:
##
##
##      b = Ts[n] - sum_i Ls[i] * Ts[i] * K(Xs[i],Xs[n])
##
##  because support vectors lie on the margin hyperplanes, hence
##  Ts[n]*y(Xs[n]) == 1 provided that Ls[n] != 0, where y(x) is the
##  discriminant function.
##
##                         Ts[n]*y(Xs[n]) == 1
##                               y(Xs[n]) == Ts[n]   (Ts[n]*Ts[n] == 1)
##                   dot(Ws,Xs)*Xs[n] + b == Ts[n]   (def of y(x))
##  (sum_i Ls[i]*Ts[i]*K(Xs[i],Xs[n])) + b == Ts[n]  (def of Ws -- weights)
##
##
##  It's numerically more stable to average over all support vectors.
##
##  N.B.  If no multipliers are supplied, this routine will call
##  makeLambdas to generate them.  If this fails, it will throw an
##  exception.
##
##  If no kernel is supplied, this routine will use the default
##  polynomial kernel  K(x,y) = (dot(x,y) + 1.0)^2.
##
##

def makeB(Xs, Ts, Ls=None, K=rbfKernel):
    "Generate the bias given Xs, Ts and (optionally) Ls and K"
    ## No Lagrange multipliers supplied, generate them.
    if Ls == None:
        status, Ls = makeLambdas(Xs, Ts)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    sv_count = 0
    b_sum = 0.0
    for n in range(len(Ts)):
        if Ls[n] >= 1e-10:  ## 1e-10 for numerical stability.
            sv_count += 1
            b_sum += Ts[n]
            for i in range(len(Ts)):
                if Ls[i] >= 1e-10:
                    b_sum -= Ls[i] * Ts[i] * K(Xs[i], Xs[n])

    return b_sum / sv_count



##--------------------------------------------------------------------------
##
##  Classify a single input vector using a trained nonlinear SVM.
##
##  If Lagrange multipliers not supplied, will build them from the
##  training set.  Ditto for bias.  By default uses a quadratic polynomial
##  kernel.  Takes a parameter, verbose, that prints out the input,
##  activation level and classification if set to True.
##
##  If no kernel is supplied, this routine will use the default
##  polynomial kernel  K(x,y) = (dot(x,y) + 1.0)^2.
##
##
##  NOTE: Set Verbose to True in order to print all classifications to the
##  console.

def classify(x, Xs, Ts, Ls=None, b=None, K=rbfKernel, verbose=False):
    "Classify an input x into {-1,+1} given support vectors, outputs and L."
    ## No Lagrange multipliers supplied, generate them.
    if Ls == None:
        status, Ls = makeLambdas(Xs, Ts)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    if b == None:  b = makeB(Xs, Ts, Ls, K)
    ## Do classification.  y is the "activation level".
    y = b
    for n in range(len(Ts)):
        if Ls[n] >= 1e-10:
            y += Ls[n] * Ts[n] * K(Xs[n], x)

    if verbose:
        print "%s %8.5f  --> " % (x, y),
        if y > 0.0:
            print "+1"
        elif y < 0.0:
            print "-1"
        else:
            print "0  (ERROR)"
	
	## Return classification results, either +1 or -1
    if y > 0.0:
        return +1
    elif y < 0.0:
        return -1
    else:
        return 0


        

        
##--------------------------------------------------------------------------
##
##  Test a trained nonlinear SVM on all vectors from its training set.  The
##  kernel is quadratic polynomial by default.
##
##  If Lagrange multipliers not supplied, will build them from the
##  training set.  Ditto for bias.  By default uses a quadratic polynomial
##  kernel.  Takes a parameter, verbose, that prints out the input,
##  activation level and classification if set to True.
##
##  If no kernel is supplied, this routine will use the default
##  polynomial kernel  K(x,y) = (dot(x,y) + 1.0)^2.
##
##   
##  NOTE: Set Verbose = True in order to print misclassification notifications
##  to the console.   
        

def testClassifier(Xs, Ts, Ls, b=None, K=rbfKernel, verbose=False):
    "Test a classifier specifed by Lagrange mults, bias and kernel on all Xs/Ts pairs."
    assert len(Xs) == len(Ts)
    assert len(Xs[0]) == 2
    
    
    ## No Ls supplied, generate them.
    if Ls == None:
        status, Ls = makeLambdas(Xs,Ts,C,K=rbfKernel)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
        print "Lagrange multipliers:", Ls
        
        
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    if b == None:
        b = makeB(Xs,Ts,C,Ls,K=rbfKernel)
        print "Bias:", b
    
    ## Do classification test.
	## Create variable to count number of misclassifications
    misclassificationCount = 0
    good = True
    for i in range(len(Xs)):
        c = classify(Xs[i],Xs,Ts,Ls,b,K=K)
        if c != Ts[i]:
            misclassificationCount += 1
            if verbose:
                print "Misclassification: input %s, output %d, expected %d" %\
                    (Xs[i],c,Ts[i])
            good = False
    
    print "Misclassification count: ", misclassificationCount 
    classificationAccuracy = misclassificationCount*100/len(Ts)
    print "Misclassification Accuracy(%): ", classificationAccuracy 	
    #Generate range of 'activation levels' from SVM
    #Sample in x and y from -5.0 to 5.0 with a resolution of 0.1
    xs = np.arange(-5, 5, 0.1)
    ys = np.arange(-5, 5, 0.1)
    als = np.zeros((len(xs), len(ys)))
    
    ## Create als vector to be fed into contour() function
	## This models the function: Sigma[ Lambda_r *t_r * K(x_r, x_c) + b]
	## If result is >0 --> point is +1
	## else --> point is -1
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            testvector = [x,y]
            al = b
            for n in range(len(Ts)):
                if Ls[n] >= 1e-10:
                    al += Ls[n] * Ts[n] * K(Xs[n],testvector)
            als[j, i] = al
    
	## Contour function needs a meshgrid to create plot
    X, Y = np.meshgrid(xs, ys)

	## Functions to plot contour at -1, 0 and +1 levels in 
	## blue, green and red. Add title with C parameter.
    plt.figure()
    plt.title('Classifier Output, C = ' + str(C) + \
              ',$\sigma^2$ = 0.25')

    plt.contour(X, Y, als, levels=[-1.0, 0.0, 1.0], linewidths=(1, 2, 1), colors=('blue', 'green', 'red'))

	## Plot points, if/else decides whether the points are
	## blue (-1) or red (+1)
    for i, t in enumerate(Ts):
        if t < 0:
            col = 'blue'		## -1 values
        else:
            col = 'red'			## +1 values
        plt.plot([Xs2[i][0]], [Xs2[i][1]], marker='o', color=col)

    plt.show()



## Execute makeLamdas and makeB functions and assign them to 
## appropriate variables
## Feed these variables to testClassifier function in order to 
## classify and plot dataset with classification boundaries
## and margins
status,Ls = makeLambdas(Xs2,Ts2,K=rbfKernel)
b = makeB(Xs2,Ts2,Ls,K=rbfKernel)
##classify(Xs2, Ts2, Ls, b, K=rbfKernel)
testClassifier(Xs2,Ts2,Ls,b, K=rbfKernel)
