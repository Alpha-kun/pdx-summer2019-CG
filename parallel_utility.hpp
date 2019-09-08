#include <omp.h>
#include <assert.h>
#include "linalgcpp.hpp"

using namespace linalgcpp;


//==================================SparseMatrix-Vector=======================================================

template <typename T>
Vector<double> ParaMult(const SparseMatrix<T>& A, const Vector<double>& b){
	int M = A.Rows();
	
	const std::vector<int>& indptr = A.GetIndptr();
	const std::vector<int>& indices = A.GetIndices();
	const std::vector<T>& data = A.GetData();
	
	Vector<double> Ab(M);
	
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		double sum=0.0;
		for(int j=indptr[i];j<indptr[i+1];j++){
			sum+=data[j]*b[indices[j]];
		}
		Ab[i]=sum;
	}
	return Ab;
}

//==================================DenseMatrix-Vector=======================================

Vector<double> ParaMult(const DenseMatrix& A, const Vector<double>& b){
	int M=A.Rows();
	int N=A.Cols();
	Vector<double> Ab(M);
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		double sum=0.0;
		for(int j=0;j<N;j++){
			sum+=A(i,j)*b[j];
		}
		Ab[i]=sum;
	}
	return Ab;
}

//===============SparseMaxtrix-Matrix=====================



template <typename U, typename V>
SparseMatrix<double> ParaMult(const SparseMatrix<U>& lhs, const SparseMatrix<V>& rhs)
{
    assert(rhs.Rows() == lhs.Cols());
	
	int cols_=lhs.Cols();
	int rows_=lhs.Rows();
	
	const std::vector<int>& indptr_=lhs.GetIndptr();
    const std::vector<int>& indices_=lhs.GetIndices();
	const std::vector<U>& data_=lhs.GetData();
	
	const std::vector<int>& rhs_indptr = rhs.GetIndptr();
    const std::vector<int>& rhs_indices = rhs.GetIndices();
    const std::vector<V>& rhs_data = rhs.GetData();
	
	std::vector<int> out_indptr(rows_ + 1);
    out_indptr[0] = 0;

    omp_set_num_threads(8);

	#pragma omp parallel
	{

	//printf("using %d threads\n",omp_get_num_threads());
	std::vector<int> marker(rhs.Cols());
    std::fill(begin(marker), end(marker), -1);
	#pragma omp for
    for (int i = 0; i < rows_; ++i){
		int row_nnz=0;
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j){
			for (int k = rhs_indptr[indices_[j]]; k < rhs_indptr[indices_[j] + 1]; ++k){
				if (marker[rhs_indices[k]] != static_cast<int>(i)){//<=================WHY static_cast?
					marker[rhs_indices[k]] = i;
					++row_nnz;
				}
			}
		}
		out_indptr[i + 1] = row_nnz;
	}
	
	}
	
	for(int i = 0; i < rows_; ++i){
		out_indptr[i + 1]+=out_indptr[i];
	}

    
    std::vector<int> out_indices(out_indptr[rows_]);
    std::vector<double> out_data(out_indptr[rows_]);
	
	
	#pragma omp parallel
	{
	
	std::vector<int> marker(rhs.Cols());
	std::fill(begin(marker), end(marker), -1);
    
	int total = 0;
	int zero_ptr = -1;
	#pragma omp for
    for (int i = 0; i < rows_; ++i)
    {
        int row_nnz = total;
		// at this point, all entries in marker <= row_nnz
		if(zero_ptr==-1){
			zero_ptr=out_indptr[i];
		}
		
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            for (int k = rhs_indptr[indices_[j]]; k < rhs_indptr[indices_[j] + 1]; ++k)
            {
                if (marker[rhs_indices[k]] < row_nnz)
                {
                    marker[rhs_indices[k]] = total;
                    out_indices[zero_ptr+total] = rhs_indices[k];
                    out_data[zero_ptr+total] = data_[j] * rhs_data[k];

                    total++;
                }
                else
                {
                    out_data[zero_ptr+marker[rhs_indices[k]]] += data_[j] * rhs_data[k];
                }
            }
        }
    }
	
	}
    return SparseMatrix<double>(std::move(out_indptr), std::move(out_indices), std::move(out_data),
                            rows_, rhs.Cols());
}
