#include "linalgcpp.hpp"
#include "partition.hpp"
#include "graphIO.hpp"
#include "condugate_gradient.hpp"
#include <cmath>
#include <chrono>
#include <ctime>

using namespace linalgcpp;


void parallel_test(const SparseMatrix<double>& R, const Vector<double>& b, int num_thrd){
	
	omp_set_num_threads(num_thrd);

	int n=R.Cols();

	#pragma omp parallel
	{
		std::cout<<"thread "<<omp_get_thread_num()<<" reporting"<<std::endl;
	}
	

	double ini_time;
	double end_time;
	std::cout << std::setw(7) << "solver" << std::setw(20) << "iter" << std::setw(15) << "time" << std::endl << std::endl;

	std::cout << std::setw(7) << "CG"; 
	ini_time = omp_get_wtime();
    CG(R,b,n,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;

	std::cout << std::setw(7) << "jacobi"; 
	ini_time = omp_get_wtime();
    PCG(R,b,Solve_Jacobian,n,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
	std::cout << std::setw(7) << "GG"; 
	ini_time = omp_get_wtime();
    PCG(R,b,Solve_Gauss_Seidel,n,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;

	std::cout << std::setw(7);
	std::cout << std::setw(7) << "TL"; 
	ini_time = omp_get_wtime();
    PCG_TL(R, b,n,1e-9,std::cbrt(R.Cols()));
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;

/**
	std::cout << std::setw(7) << "ML"; 
	ini_time = omp_get_wtime();
    PCG_ML(R,b,n,1e-9,20,std::cbrt(R.Cols()));
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
*/	
}

int main()
{
    
	
	SparseMatrix<double> R = ReadMTX("/data/sparse/tmt_sym.mtx");
	int n = R.Cols();
	
	//generate right-hand-side
	Vector<double> b=RandVect(n,10000);

	for(int i=2;i<=20;i+=2){
		parallel_test(R,b,i);
	}
	
	
	
}
