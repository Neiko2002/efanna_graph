//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_kdtree.h>
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


int main(int argc, char** argv) {

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

  // auto object_file      = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
  // auto graph_filename   = R"(e:/Data/Feature/SIFT1M/efanna/efanna_kdtree_nTree8_mLevel8_K60.kdt)";

  auto object_file      = R"(e:/Data/Feature/GloVe/glove-100/glove-100_base.fvecs)";
  auto graph_filename   = R"(e:/Data/Feature/GloVe/efanna/efanna_kdtree_nTree8_mLevel8_K100.kdt)";

  // WEAVESS sift1M
  // unsigned nTrees = (unsigned)8;
  // unsigned mLevel = (unsigned)8;
  // unsigned K = (unsigned)60;

   // WEAVESS glove
  unsigned nTrees = (unsigned)8;
  unsigned mLevel = (unsigned)8;
  unsigned K = (unsigned)100;

  std::cout << "Load Data" << std::endl;
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(object_file, data_load, points_num, dim);
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading data" << std::endl;

  std::cout << "Align Data" << std::endl;
  data_load = efanna2e::data_align(data_load, points_num, dim); //one must align the data before build
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after aligning data" << std::endl;

  std::cout << "Create kdtree" << std::endl;
  efanna2e::IndexKDtree index(dim, points_num, efanna2e::L2, nullptr);
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after creating kdtree" << std::endl;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("K", K);
  paras.Set<unsigned>("nTrees", nTrees);
  paras.Set<unsigned>("mLevel", mLevel);


  std::cout << "Build kdtree" << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e-s;
  std::cout <<"Time cost: "<< diff.count() << "\n";
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after building kdtree" << std::endl;



  index.Save(graph_filename);

  return 0;
}
