/*! @file

    @brief A collection of condugate gradient and preconditioned condugate gradient methods.
*/


#include <assert.h>
#include "linalgcpp.hpp"
#include "parallel_utility.hpp"
#include "lubys_partition.hpp"

using namespace linalgcpp;


inline
Vector<double> entrywise_mult(const Vector<double>& a, const Vector<double>& b){
	assert (a.size()==b.size());
	
	Vector<double> c(a.size());
	for(int k=0;k<a.size();++k){
		c[k]=a[k]*b[k];
	}
	return c;
}

Vector<double> entrywise_inv(const Vector<double>& a){
	
	Vector<double> c(a.size());
	for(int k=0;k<a.size();++k){
		c[k]=1.0/a[k];
	}
	return c;
}

//M=D+L, the lower triangular system (forward Gauss-Seidel)
Vector<double> DLsolver(const SparseMatrix<double>& M, Vector<double> b){
	assert(M.Rows()==M.Cols()&&M.Rows()==b.size());
	
	const std::vector<int>& indptr=M.GetIndptr();
	const std::vector<int>& indices=M.GetIndices();
	const std::vector<double>& data=M.GetData();

	for(int i=0;i<M.Rows();++i){
		double sum=0.0;
		double pivot;
		for(int j=indptr[i];j<indptr[i+1];++j){
			if(indices[j]<i){
				sum+=b[indices[j]]*data[j];
			}
			if(indices[j]==i){
				pivot=data[j];
			}
		}
		b[i]-=sum;
		b[i]/=pivot;
	}
	return b;
}



//solve the upper triangular system (backward Gauss-Seidel)
Vector<double> DUsolver(const SparseMatrix<double>& MT, Vector<double> b){
	assert(MT.Rows()==MT.Cols()&&MT.Rows()==b.size());
	
	const std::vector<int>& indptr=MT.GetIndptr();
	const std::vector<int>& indices=MT.GetIndices();
	const std::vector<double>& data=MT.GetData();
	
	for(int i= MT.Rows()-1;i>=0;--i){
		double sum=0;
		double pivot;
		for(int j=indptr[i];j<indptr[i+1];++j){
			if(indices[j]>i){
				sum+=b[indices[j]]*data[j];
			}
			if(indices[j]==i){
				pivot=data[j];
			}
		}
		b[i]-=sum;
		b[i]/=pivot;
	}
	return b;
}

inline
Vector<double> Solve_TL(const SparseMatrix<double>& A,
						const DenseMatrix& Ac, 
						const SparseMatrix<int>& P, 
						const DenseMatrix& AcInverse,
						Vector<double>(*Msolver)(const SparseMatrix<double>& , Vector<double>),
						Vector<double>(*MTsolver)(const SparseMatrix<double>& , Vector<double>),
						const Vector<double>& b){
	
	//1: "Pre-smooth" solve for x1/3
	Vector<double> x13 = Msolver(A,b);
	//2: compute restrictive residual
	Vector<double> rc = P.MultAT(b-A.Mult(x13));
	//3: solve for xc
	Vector<double> xc = AcInverse.Mult(rc);
	//4: fine-level approximation
	Vector<double> x23 = x13+Mult(P,xc);
	//5: compute and return x
	return MTsolver(A,b-A.Mult(x23))+x23;
}

/*! @brief The two-level condugate gradient method

    @param A0 an s.p.d. matrix
    @param b the right-hand-side of system
    @param max_iter maximum number of iteration before exit
	@param tol epsilon, the relative error tolerence
	@param Ncoarse the dimension of the coarse matrix
*/
Vector<double> PCG_TL(const SparseMatrix<double>& A, const Vector<double>& b,int max_iter,double tol,int Ncoarse){
	//assert A is s.p.d.
	assert(1<=Ncoarse && Ncoarse< A.Cols());
	
	
	Vector<int> partitions = Partition(A,Ncoarse);
	SparseMatrix<int> P = Unweighted(partitions);
	
	DenseMatrix Ac = P.Transpose().Mult(A.Mult(P)).ToDense();
	
	DenseMatrix AcInverse;
	Ac.Invert(AcInverse);

	int n = A.Cols();
	
	Vector<double> x(n,0.0);
	Vector<double> r(b);
	Vector<double> pr = Solve_TL(A,Ac,P,AcInverse,*DLsolver,*DUsolver,r);
	Vector<double> p(pr);
	Vector<double> g(n);
	double delta0 = r.Mult(pr);
	double delta = delta0, deltaOld, tau, alpha;

	for(int k=0;k<max_iter;k++){
		g = A.Mult(p);
		tau = p.Mult(g);
		alpha = delta / tau;
		x = x + (alpha * p);
		r = r - (alpha * g);
		pr = Solve_TL(A,Ac,P,AcInverse,*DLsolver,*DUsolver,r);
		deltaOld = delta;
		delta = r.Mult(pr);
        if(delta < tol * tol * delta0){
			std::cout<<"converge at iteration "<<k<<std::endl;
			return x;
        }
	   	p = pr + ((delta / deltaOld)* p);
    }
	
	std::cout<<"failed to converge in "<<max_iter<<" iterations"<<std::endl;
	return x;
	
}

Vector<double> Solve_ML(const std::vector<SparseMatrix<double>>& A,
						const std::vector<SparseMatrix<int>>& P,
						const DenseMatrix& AcInverse,
						Vector<double>(*Msolver)(const SparseMatrix<double>& , Vector<double>),
						Vector<double>(*MTsolver)(const SparseMatrix<double>& , Vector<double>),
						const Vector<double>& b){
							
	int L = A.size() - 1;
	std::vector<Vector<double>> r(L+1);
	std::vector<Vector<double>> x(L+1);
	r[0]=b;
	
	for(int i=0;i<L;i++){
		x[i]=Msolver(A[i],r[i]);
		r[i+1]=P[i].MultAT(r[i]-A[i].Mult(x[i]));
	}
	
	x[L]= AcInverse.Mult(r[L]);
	
	for(int i=L-1;i>=0;i--){
		x[i]=x[i]+P[i].Mult(x[i+1]);
		x[i]=x[i]+MTsolver(A[i],r[i]-A[i].Mult(x[i]));
	}
	
	return x[0];
}

/*! @brief The multi-level condugate gradient method

    @param A0 an s.p.d. matrix
    @param b the right-hand-side of system
    @param max_iter maximum number of iteration before exit
	@param tol epsilon, the relative error tolerence
	@param Lmax the maximum number of levels
	@param Ncoarse the dimension of the coarsest matrix
*/
Vector<double> PCG_ML(const SparseMatrix<double>& A0, const Vector<double>& b,int max_iter,double tol, int Lmax, int Ncoarse){
	//assert A0 is s.p.d.
	assert(Lmax >= 1);
	assert (A0.Cols() >= Ncoarse && Ncoarse >= 1);
	
	std::vector<int> N(Lmax+1);
	N[0]=A0.Cols();
	
	double q = std::min(pow(1.0*Ncoarse/N[0],1.0/Lmax),0.5);
	
	int L = 0;// the index of last Nk
	for(int i = 1; i<Lmax;i++){
		N[i]=N[i-1]*q;
		if(N[i]!=0) L++;
		if(N[i]<=Ncoarse) break;
	}
	
	std::vector<SparseMatrix<int>> P(L);
	std::vector<SparseMatrix<double>> A(L+1);
	A[0]=A0;
	
	for(int i=0;i<L;i++){
		P[i] = getP_rand(A[i],N[i+1]);
		A[i+1]= P[i].Transpose().Mult(A[i].Mult(P[i]));
	}
	
	DenseMatrix AcInverse;
	A[L].ToDense().Invert(AcInverse);
	

    Vector<double> x(N[0],0.0);
    Vector<double> r(b);
	Vector<double> pr = Solve_ML(A,P,AcInverse,*DLsolver,*DUsolver,r);
    Vector<double> p(pr);
    Vector<double> g(N[0]);
    double delta0 = r.Mult(pr);
    double delta = delta0, deltaOld, tau, alpha;

    for(int k=0;k<max_iter;k++){
        g = A[0].Mult(p);
        tau = p.Mult(g);
        alpha = delta / tau;
        x = x + (alpha * p);
        r = r - (alpha * g);
        pr = Solve_ML(A,P,AcInverse,*DLsolver,*DUsolver,r);
		deltaOld = delta;
		delta = r.Mult(pr);
        if(delta < tol * tol * delta0){
            std::cout<<"converge at iteration "<<k<<std::endl;
            return x;
        }
        p = pr + ((delta / deltaOld)* p);
    }
	
    std::cout<<"failed to converge in "<<max_iter<<" iterations"<<std::endl;
    return x;
	
}



//used by both jacobi and gauss-seidel
Vector<double> diag;

/*! @brief Solve system Mx=r, where M is the symmetric Gauss-Seidel matrix of A, in O(# of non-zero entries in A)

    @param A the sparse maxtrix from which we generate M
    @param r the right-hand-side of the system
*/
inline
Vector<double> Solve_Gauss_Seidel(const SparseMatrix<double>& A, Vector<double> r){
	//assert(A.Rows()==A.Cols()&&A.Rows()==r.size());
	
	//step 1: solve the lower triangular system for y: (D+L)y=r
	r=DLsolver(A,r);
	//step 2: solve the upper triangular system for x: (D+U)x=Dy
	r=entrywise_mult(diag,r);
	return DUsolver(A,r);
	
}

inline
Vector<double> Solve_Jacobian(const SparseMatrix<double>& A, Vector<double> r){
	//assert(A.Rows()==A.Cols()&&A.Rows()==r.size());
	
	int n = A.Cols();
	for(int i=0;i<n;++i){
		r[i]/=diag[i];
	}
	return r;
}

/*! @brief The preconditioned condugate gradient method

    @param A an s.p.d. matrix
    @param b the right-hand-side of system
	@param Msolver the preconditioning function which solves Mx=b for a particular preconditioner M
    @param max_iter maximum number of iteration before exit
	@param tol epsilon
*/
Vector<double> PCG(const SparseMatrix<double>& A, const Vector<double>& b, Vector<double>(*Msolver)(const SparseMatrix<double>& , Vector<double>),int max_iter,double tol){
	//assert A is s.p.d.
	
    int n = A.Cols();
	
	//set diag for quick access from jacobi and gauss-seidel
	diag = Vector<double>(&A.GetDiag()[0],n);
    
	Vector<double> x(n,0.0);
    Vector<double> r(b);
	Vector<double> pr = Msolver(A,r);
    Vector<double> p(pr);
    Vector<double> g(n);
    double delta0 = r.Mult(pr);
    double delta = delta0, deltaOld, tau, alpha;
	
	
    for(int k=0;k<max_iter;k++){
        g = A.Mult(p);
        tau = p.Mult(g);
        alpha = delta / tau;
        x = x + (alpha * p);
        r = r - (alpha * g);
        pr = Msolver(A,r);
		deltaOld = delta;
		delta = r.Mult(pr);
        if(delta < tol * tol * delta0){
            std::cout<<"converge at iteration "<<k<<std::endl;
            return x;
        }
        p = pr + ((delta / deltaOld)* p);
    }
	
    std::cout<<"failed to converge in "<<max_iter<<" iterations"<<std::endl;
    return x;
	
}

/*! @brief The regular condugate gradient method, time complexity O(max_iter*N^2)

    @param A an s.p.d. matrix
    @param b the right-hand-side of system
    @param max_iter maximum number of iteration before exit
	@param tol epsilon
*/
Vector<double> CG(const SparseMatrix<double>& A, const Vector<double>& b, int max_iter,double tol){
    //assert A is s.p.d.
    int n = A.Cols();
    Vector<double> x(n,0.0);
    Vector<double> r(b);
    Vector<double> p(r);
    Vector<double> g(n);
    double delta0 = b.Mult(b);
    double delta = delta0, deltaOld, tau, alpha;

    for(int k=0;k<max_iter;k++){
		g = A.Mult(p);
        tau = p.Mult(g);
        alpha = delta / tau;
        x = x + (alpha * p);
        r = r - (alpha * g);
        deltaOld = delta;
		delta = r.Mult(r);
        if(delta < tol * tol * delta0){
            std::cout<<"converge at iteration "<<k<<std::endl;
            return x;
        }
        p = r + ((delta / deltaOld)* p);
    }
	
    std::cout<<"failed to converge in "<<max_iter<<" iterations"<<std::endl;
    return x;
}

