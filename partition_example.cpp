#include "linalgcpp.hpp"
#include "partition.hpp"
#include "graphIO.hpp"
#include "conjugate_gradient.hpp"
#include <cmath>
#include <chrono>
#include <ctime>

using namespace linalgcpp;



void solver_test(){
	//get matrix 
	SparseMatrix<double> R = ReadMTX("/data/sparse/tmt_sym.mtx");
	
	int n = R.Cols();
	
	//generate right-hand-side
	//Vector<double> b=RandVect(R.Cols(),10000);
	Vector<double> b(ReadText("/data/tmt.txt"));
		
	double ini_time;
	double end_time;
	std::cout << std::setw(7) << "solver" << std::setw(20) << "iter" << std::setw(15) << "time" << std::endl << std::endl;

//std::cout<<'#'<<std::endl;
	std::cout << std::setw(7) << "CG"; 
	ini_time = omp_get_wtime();
    CG(R,b,n,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
//std::cout<<'#'<<std::endl;
	std::cout << std::setw(7) << "jacobi"; 
	ini_time = omp_get_wtime();
    PCG(R,b,Solve_Jacobian,n,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
//std::cout<<'#'<<std::endl;
	std::cout << std::setw(7) << "GS"; 
	ini_time = omp_get_wtime();
    PCG(R,b,Solve_Gauss_Seidel,n,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;

//std::cout<<'#'<<std::endl;
	std::cout << std::setw(7) << "TL"; 
	ini_time = omp_get_wtime();
    PCG_TL(R, b,n,1e-9,std::cbrt(R.Cols()));
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;

//std::cout<<'#'<<std::endl;
	std::cout << std::setw(7) << "ML"; 
	ini_time = omp_get_wtime();
    PCG_ML(R,b,n,1e-9,20,std::cbrt(R.Cols()));
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
}

void solver_test(const SparseMatrix<double>& R){
	
	//generate right-hand-side
	Vector<double> b=RandVect(R.Cols(),10000);
	
	std::cout<<"dimension of A: "<<R.Cols()<<std::endl; 
	
	double ini_time;
	double end_time;
	std::cout << std::setw(7) << "solver" << std::setw(20) << "iter" << std::setw(15) << "time" << std::endl << std::endl;
	
	std::cout << std::setw(7) << "CG"; 
	ini_time = omp_get_wtime();
    CG(R,b,100000,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
	std::cout << std::setw(7) << "jacobi"; 
	ini_time = omp_get_wtime();
    PCG(R,b,Solve_Jacobian,100000,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
	std::cout << std::setw(7) << "GS"; 
	ini_time = omp_get_wtime();
    PCG(R,b,Solve_Gauss_Seidel,100000,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;

	std::cout << std::setw(7) << "TL"; 
	ini_time = omp_get_wtime();
    PCG_TL(R, b,100000,1e-9,std::cbrt(R.Cols()));
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;

	std::cout << std::setw(7) << "ML"; 
	ini_time = omp_get_wtime();
    PCG_ML(R,b,100000,1e-9,20,std::cbrt(R.Cols()));
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
}

int main()
{
    
	
	SparseMatrix<double> Laplacian = getLaplacian("/sc-pwtk.mtx",1,false);
	SparseMatrix<double> RLap = getReducedLaplacian(Laplacian);
	
	//SparseMatrix<double> R = ReadMTX("/home/ruofeng/cluster/data/sparse/tmt_sym.mtx");
	//std::cout<<"testing partition"<<std::endl;
	
	//std::vector<double> vb=ReadText("/home/ruofeng/cluster/data/matlabData/b1.txt");
	//Vector<double> b(vb);
	//double ini_time;
	//double end_time;

	solver_test();
	solver_test(R);
	
	
	//Vector<double> b=RandVect(R.Cols(),10000);
	//PCG_ML(R,b,10000,1e-9,100,std::cbrt(R.Cols()));
	
}
