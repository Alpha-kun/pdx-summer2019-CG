#define watch(x) std::cout << (#x) << " is " << (x) << std::endl
#define log(x) std::cout << x << std::endl
#include "linalgcpp.hpp"
#include <omp.h>
#include "randomGen.hpp"

using namespace linalgcpp;


void print(std::pair<int,int> pr){
	std::cout<<pr.first<<","<<pr.second<<std::endl;
}

SparseMatrix<int> getP(const SparseMatrix<double>& A){
	int n = A.Cols();
	std::vector<std::pair<int,int>> max_edge(n);
	
	
	const std::vector<int>& indptr=A.GetIndptr();
	const std::vector<int>& indices=A.GetIndices();
	const std::vector<double>& data=A.GetData();
	
	for(int i=0;i<n;i++){
		int max_index = indices[indptr[i]];
		double max_weight = data[indptr[i]];
		
		for(int j=indptr[i];j<indptr[i+1];++j){
			if(indices[j]==i) continue;
			if(data[j]>max_weight){
				max_index=indices[j];
				max_weight=data[j];
			}
		}
		max_edge[i]=std::make_pair(std::min(i,max_index),std::max(i,max_index));
	}
	
	CooMatrix<int> P;
	std::vector<bool> registered(n);
	int col=0;
	for(int i=0;i<n;i++){
		if(registered[i]) continue;
		if(max_edge[i].second!=i&&max_edge[max_edge[i].second]==max_edge[i]){ 
			P.Add(max_edge[i].first,col,1);
			P.Add(max_edge[i].second,col++,1);
			registered[max_edge[i].second] =true;
		}else{
			P.Add(i,col++,1);
		}
	}
	
	return P.ToSparse();
	
}

SparseMatrix<int> getP(const SparseMatrix<double>& A, int Ncoarse){
	//assert Ncoarse<A.cols()
	
	int n = A.Cols();
	
	std::vector<std::pair<int,int>> max_edge(n);
	
	const std::vector<int>& indptr=A.GetIndptr();
	const std::vector<int>& indices=A.GetIndices();
	const std::vector<double>& data=A.GetData();
	
	for(int i=0;i<n;i++){
		int max_index = indices[indptr[i]];
		double max_weight = data[indptr[i]];
		
		for(int j=indptr[i];j<indptr[i+1];++j){
			if(indices[j]==i) continue;
			if(data[j]>max_weight){
				max_index=indices[j];
				max_weight=data[j];
			}
		}
		max_edge[i]=std::make_pair(std::min(i,max_index),std::max(i,max_index));
	}
	
	CooMatrix<int> P;
	std::vector<bool> registered(n);
	int col=0;
	for(int i=0;i<n;i++){
		if(registered[i]) continue;
		if(max_edge[i].second!=i&&max_edge[max_edge[i].second]==max_edge[i]){ 
			P.Add(max_edge[i].first,col,1);
			P.Add(max_edge[i].second,col++,1);
			registered[max_edge[i].second] = true;
		}else{
			P.Add(i,col++,1);
		}
	}
	
	
	SparseMatrix<int> Ps = P.ToSparse();
	
	if(col==Ncoarse){
		std::cout<<"warning: the matrix is diagonal, no edge to contract"<<std::endl;
		return Ps;
	}
	
	if(col<=Ncoarse){
		return Ps;
	}else{
		return Ps.Mult(getP(Ps.Transpose().Mult(A.Mult(Ps)),Ncoarse));
	}
	
}

SparseMatrix<int> getP_rand(const SparseMatrix<double>& A, int Ncoarse){
	//assert Ncoarse<A.cols()
	
	int n = A.Cols();
	
	std::vector<std::pair<int,int>> max_edge(n);
	
	const std::vector<int>& indptr_=A.GetIndptr();
	const std::vector<int>& indices_=A.GetIndices();
	
	CooMatrix<double> Rand(n);
	
	for(int i=0;i<n;i++){
		for(int j=indptr_[i];j<indptr_[i+1];j++){
			if(indices_[j]<i){
				Rand.AddSym(i,indices_[j],rand());
			}
			if(indices_[j]==i){
				Rand.Add(i,indices_[j],rand());
			}
		}
	}
	
	SparseMatrix<double> SR = Rand.ToSparse();
	const std::vector<int>& indptr=SR.GetIndptr();
	const std::vector<int>& indices=SR.GetIndices();
	const std::vector<double>& data=SR.GetData();
	
	#pragma omp parallel for
	for(int i=0;i<n;i++){
		if(indptr[i]+1==indptr[i+1]&&indices[indptr[i]]==i){
			max_edge[i]=std::make_pair(i,i);
			continue;
		}
		int max_index = 0;
		double max_weight = -1.0;
		for(int j=indptr[i];j<indptr[i+1];++j){
			if(indices[j]==i) continue;
			if(data[j]>max_weight){
				max_index=indices[j];
				max_weight=data[j];
			}
		}
		max_edge[i]=std::make_pair(std::min(i,max_index),std::max(i,max_index));
	}
	
	CooMatrix<int> P;
	std::vector<bool> registered(n);
	int col=0;
	for(int i=0;i<n;i++){
		if(registered[i]) continue;
		if(max_edge[i].second!=i&&max_edge[max_edge[i].second]==max_edge[i]){ 
			P.Add(max_edge[i].first,col,1);
			P.Add(max_edge[i].second,col++,1);
			registered[max_edge[i].second] = true;
		}else{
			P.Add(i,col++,1);
		}
	}
	SparseMatrix<int> Ps = P.ToSparse();
	
	if(col==n){
		std::cout<<"warning: the matrix is diagonal, no edge to contract"<<std::endl;
		return Ps;
	}
	
	if(col<=Ncoarse){
		return Ps;
	}else{
		return Ps.Mult(getP_rand(Ps.Transpose().Mult(A.Mult(Ps)),Ncoarse));
	}
	
}
