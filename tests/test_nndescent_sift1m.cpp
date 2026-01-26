//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <unordered_set>
#include <cmath>

void load_data(const char* filename, float*& data, unsigned& num, unsigned& dim) { // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
  in.read((char*)&dim,4);
  std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0,std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim+1) / 4);
  data = new float[num * dim * sizeof(float)];

  in.seekg(0,std::ios::beg);
  for(size_t i = 0; i < num; i++){
    in.seekg(4,std::ios::cur);
    in.read((char*)(data+i*dim),dim*4);
  }
  in.close();
}

int main(int argc, char** argv){

  #if defined(__AVX__)
    std::cout << "use AVX2  ..." << std::endl;
  #elif defined(__SSE2__)
    std::cout << "use SSE  ..." << std::endl;
  #else
    std::cout << "use arch  ..." << std::endl;
  #endif
  std::cout << "DATA_ALIGN_FACTOR " << DATA_ALIGN_FACTOR << std::endl;

  #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
  #endif


  auto object_file      = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
  auto efanna_file      = R"(e:/Data/Feature/SIFT1M/efanna/efanna K50 L70 It10 S10 R50 again.efa)";
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb" << std::endl;

  std::cout << "Load Data" << std::endl;
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(object_file, data_load, points_num, dim);
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading data" << std::endl;

  

  // Efanna: test_nndescent sift_base.fvecs sift.50NN.graph 50 70 10 10 50
  // https://github.com/Neiko2002/efanna_graph
  unsigned K = (unsigned)50;
  unsigned L = (unsigned)70;
  unsigned iter = (unsigned)10;
  unsigned S = (unsigned)10;
  unsigned R = (unsigned)50;

  // NSG: test_nndescent sift.fvecs sift_200nn.graph 200 200 10 10 100  
  // https://github.com/Neiko2002/nsg
  // unsigned K = (unsigned)200;
  // unsigned L = (unsigned)200;
  // unsigned iter = (unsigned)10;
  // unsigned S = (unsigned)10;
  // unsigned R = (unsigned)100;

  // SSG: test_nndescent sift.fvecs sift_200nn.knng 200 200 12 10 100
  // https://github.com/Neiko2002/SSG
  // unsigned K = (unsigned)200;
  // unsigned L = (unsigned)200;
  // unsigned iter = (unsigned)12;
  // unsigned S = (unsigned)10;
  // unsigned R = (unsigned)100;

  // WEAVES for efanna must be refined
  // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters

  // WEAVES for NSG
  // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // unsigned K = (unsigned)100;
  // unsigned L = (unsigned)120;
  // unsigned iter = (unsigned)12;
  // unsigned S = (unsigned)25;
  // unsigned R = (unsigned)300;

  // WEAVES for SSG
  // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // unsigned K = (unsigned)400;
  // unsigned L = (unsigned)420;
  // unsigned iter = (unsigned)12;
  // unsigned S = (unsigned)20;
  // unsigned R = (unsigned)100;

  // our best parameters
  // unsigned K = (unsigned)200;
  // unsigned L = (unsigned)200;
  // unsigned iter = (unsigned)20;
  // unsigned S = (unsigned)10;
  // unsigned R = (unsigned)100;
  
  std::cout << "Align Data" << std::endl;
  data_load = efanna2e::data_align(data_load, points_num, dim); // one must align the data before build
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after aligning data" << std::endl;

  std::cout << "Create graph" << std::endl;
  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after creating graph" << std::endl;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("K", K);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("iter", iter);
  paras.Set<unsigned>("S", S);
  paras.Set<unsigned>("R", R);

  std::cout << "Build graph" << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = e-s;
  std::cout <<"Time cost: "<< elapsed_time.count() << "\n";
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after building graph" << std::endl;


  index.Save(efanna_file);

  return 0;
}
