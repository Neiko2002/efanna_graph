//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_kdtree.h>
#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <unordered_set>
#include <cmath>

#include "test_util.h"




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

  // // ------------------------ SIFT1M ------------------------
  // auto object_file      = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
  // auto top_list_file    = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_base_top1000.ivecs)";
  // auto query_file       = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_groundtruth.ivecs)";
  // auto efanna_dir       = R"(e:/Data/Feature/SIFT1M/efanna/)";

  // size_t test_k = 100;
  // std::vector<unsigned> L_search_parameter = { 100, 200, 300, 500, 800, 1200, 1600 };

  // // Efanna
  // // https://github.com/Neiko2002/efanna_graph
  // unsigned nTrees = (unsigned)0;
  // unsigned mLevel = (unsigned)0;
  // unsigned K = (unsigned)50;
  // unsigned L = (unsigned)70;
  // unsigned iter = (unsigned)10;
  // unsigned S = (unsigned)10;
  // unsigned R = (unsigned)50;

  // // NSG efanna graph
  // // https://github.com/Neiko2002/nsg
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K = (unsigned)200;
  // // unsigned L = (unsigned)200;
  // // unsigned iter = (unsigned)10;
  // // unsigned S = (unsigned)10;
  // // unsigned R = (unsigned)100;

  // // SSG efanna graph 
  // // https://github.com/Neiko2002/SSG
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K = (unsigned)200;
  // // unsigned L = (unsigned)200;
  // // unsigned iter = (unsigned)12;
  // // unsigned S = (unsigned)10;
  // // unsigned R = (unsigned)100;

  // // WEAVES efanna graph must be refined
  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters

  // // WEAVES efanna graph for NSG
  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K = (unsigned)100;
  // // unsigned L = (unsigned)120;
  // // unsigned iter = (unsigned)12;
  // // unsigned S = (unsigned)25;
  // // unsigned R = (unsigned)300;

  // // WEAVES efanna graph for SSG
  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K = (unsigned)400;
  // // unsigned L = (unsigned)420;
  // // unsigned iter = (unsigned)12;
  // // unsigned S = (unsigned)20;
  // // unsigned R = (unsigned)100;

  // // our best parameters
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K = (unsigned)200;
  // // unsigned L = (unsigned)200;
  // // unsigned iter = (unsigned)20;
  // // unsigned S = (unsigned)10;
  // // unsigned R = (unsigned)100;



  // // ------------------------ GloVe ------------------------
  // auto object_file      = R"(e:/Data/Feature/GloVe/glove-100/glove-100_base.fvecs)";
  // auto top_list_file    = R"(c:/Data/Feature/GloVe/glove-100/glove_base_top1000.ivecs)";
  // auto query_file       = R"(e:/Data/Feature/GloVe/glove-100/glove-100_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/GloVe/glove-100/glove-100_groundtruth.ivecs)";
  // auto efanna_dir       = R"(e:/Data/Feature/GloVe/efanna/)";

  // size_t test_k = 100;
  // std::vector<unsigned> L_search_parameter = { 300, 800, 1000, 2000, 4000, 8000, 16000, 32000 };

  // // SSG efanna graph
  // // https://github.com/Neiko2002/SSG
  // unsigned nTrees = (unsigned)0;
  // unsigned mLevel = (unsigned)0;
  // unsigned K = (unsigned)400;
  // unsigned L = (unsigned)420;
  // unsigned iter = (unsigned)12;
  // unsigned S = (unsigned)15;     // 20 (command) or 15 (table)
  // unsigned R = (unsigned)200;

  // // WEAVES efanna graph must be refined
  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters

  // // WEAVES efanna graph for NSG
  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K = (unsigned)400;
  // // unsigned L = (unsigned)420;
  // // unsigned iter = (unsigned)12;
  // // unsigned S = (unsigned)20;
  // // unsigned R = (unsigned)300;

  // // WEAVES efanna graph for SSG
  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K = (unsigned)300;
  // // unsigned L = (unsigned)320;
  // // unsigned iter = (unsigned)12;
  // // unsigned S = (unsigned)10;
  // // unsigned R = (unsigned)200;

  // // our best parameters
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K = (unsigned)400;
  // // unsigned L = (unsigned)420;
  // // unsigned iter = (unsigned)20;
  // // unsigned S = (unsigned)20;
  // // unsigned R = (unsigned)200;



  // // ------------------------ UQ-V ------------------------
  // auto object_file      = R"(e:/Data/Feature/UQ-V/uqv/uqv_base.fvecs)";
  // auto top_list_file    = R"(e:/Data/Feature/UQ-V/uqv/uqv_base_top1000.ivecs)";
  // auto query_file       = R"(e:/Data/Feature/UQ-V/uqv/uqv_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/UQ-V/uqv/uqv_groundtruth.ivecs)";
  // auto efanna_dir       = R"(e:/Data/Feature/UQ-V/efanna/)";

  // size_t test_k = 100;
  // std::vector<unsigned> L_search_parameter = { 100, 200, 300, 500, 800, 1200, 1600 };

  // // WEAVES efanna graph for UQ-V
  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // unsigned nTrees = (unsigned)4;
  // unsigned mLevel = (unsigned)8;
  // unsigned K      = (unsigned)40;
  // unsigned L      = (unsigned)50;
  // unsigned iter   = (unsigned)7;
  // unsigned S      = (unsigned)10;
  // unsigned R      = (unsigned)150;



  // // ------------------------ enron ------------------------
  // auto object_file      = R"(e:/Data/Feature/Enron/enron/enron_base.fvecs)";
  // auto top_list_file    = R"(e:/Data/Feature/Enron/enron/enron_base_top1000.ivecs)";
  // auto query_file       = R"(e:/Data/Feature/Enron/enron/enron_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/Enron/enron/enron_groundtruth_top1000.ivecs)";
  // auto efanna_dir       = R"(e:/Data/Feature/Enron/efanna/)";

  // size_t test_k = 100;
  // size_t repeat_test = 10;
  // std::vector<unsigned> L_search_parameter = { 500, 800, 1200, 3000, 6000 };

  // // // WEAVES efanna graph for enron
  // // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // unsigned nTrees = (unsigned)4;
  // unsigned mLevel = (unsigned)8;
  // unsigned K      = (unsigned)40;
  // unsigned L      = (unsigned)140;
  // unsigned iter   = (unsigned)5;
  // unsigned S      = (unsigned)35;
  // unsigned R      = (unsigned)150;

  // // WEAVES efanna graph for SSG for enron
  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K      = (unsigned)100;
  // // unsigned L      = (unsigned)110;
  // // unsigned iter   = (unsigned)7;
  // // unsigned S      = (unsigned)20;
  // // unsigned R      = (unsigned)300;

  // // WEAVES efanna graph for NSG for enron
  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K      = (unsigned)200;
  // // unsigned L      = (unsigned)200;
  // // unsigned iter   = (unsigned)7;
  // // unsigned S      = (unsigned)25;
  // // unsigned R      = (unsigned)200;


  // // ------------------------ audio ------------------------
  // auto object_file      = R"(e:/Data/Feature/Audio/audio/audio_base.fvecs)";
  // auto top_list_file    = R"(e:/Data/Feature/Audio/audio/audio_base_top1000.ivecs)";
  // auto query_file       = R"(e:/Data/Feature/Audio/audio/audio_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/Audio/audio/audio_groundtruth_top1000.ivecs)";
  // auto efanna_dir       = R"(e:/Data/Feature/Audio/efanna/)";

  // size_t test_k = 100;
  // size_t repeat_test = 50;
  // std::vector<unsigned> L_search_parameter = { 100, 125, 180, 250, 350, 600, 1200, 2000 };

  // // WEAVES efanna graph for enron
  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // unsigned nTrees = (unsigned)16;
  // unsigned mLevel = (unsigned)8;
  // unsigned K      = (unsigned)40;
  // unsigned L      = (unsigned)40;
  // unsigned iter   = (unsigned)10;
  // unsigned S      = (unsigned)30;
  // unsigned R      = (unsigned)100;

  // // WEAVES efanna graph for SSG for enron
  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K      = (unsigned)400;
  // // unsigned L      = (unsigned)400;
  // // unsigned iter   = (unsigned)5;
  // // unsigned S      = (unsigned)25;
  // // unsigned R      = (unsigned)200;

  // // WEAVES efanna graph for NSG for enron
  // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K      = (unsigned)200;
  // // unsigned L      = (unsigned)230;
  // // unsigned iter   = (unsigned)5;
  // // unsigned S      = (unsigned)10;
  // // unsigned R      = (unsigned)100;



  // // ------------------------ imagenet1k ------------------------
  // auto object_file      = R"(e:/Data/Feature/ImageNet1k/ImageNet1k/clip_base.fvecs)";
  // auto query_file       = R"(e:/Data/Feature/ImageNet1k/ImageNet1k/clip_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/ImageNet1k/ImageNet1k/clip_groundtruth.ivecs)";
  // auto efanna_dir       = R"(e:/Data/Feature/ImageNet1k/efanna/)";

  // size_t test_k = 100;
  // size_t repeat_test = 1;
  // std::vector<unsigned> L_search_parameter = { 100, 200, 300, 500, 800, 1200, 1600 };
    
  // // NSG efanna graph for SIFT1M
  // // https://github.com/Neiko2002/nsg
  // unsigned nTrees = (unsigned)0;
  // unsigned mLevel = (unsigned)0;
  // unsigned K = (unsigned)200;
  // unsigned L = (unsigned)200;
  // unsigned iter = (unsigned)10;
  // unsigned S = (unsigned)10;
  // unsigned R = (unsigned)100;

  // // SSG efanna graph for SIFT1M
  // // https://github.com/Neiko2002/SSG
  // // unsigned nTrees = (unsigned)0;
  // // unsigned mLevel = (unsigned)0;
  // // unsigned K = (unsigned)200;
  // // unsigned L = (unsigned)200;
  // // unsigned iter = (unsigned)12;
  // // unsigned S = (unsigned)10;
  // // unsigned R = (unsigned)100;



  // // ------------------------ Deep10M ------------------------
  // auto object_file      = R"(e:/Data/Feature/Deep10M/deep10m/deep10m_base.fvecs)";
  // auto query_file       = R"(e:/Data/Feature/Deep10M/deep10m/deep10m_query.fvecs)";
  // auto groundtruth_file = R"(e:/Data/Feature/Deep10M/deep10m/deep10m_groundtruth.ivecs)";
  // auto efanna_dir       = R"(e:/Data/Feature/Deep10M/efanna/)";

  // size_t test_k = 100;
  // size_t repeat_test = 1;
  // std::vector<unsigned> L_search_parameter = { 100, 200, 300, 500, 800, 1200, 1600 };
    
  // // NSG efanna graph for SIFT1M
  // // https://github.com/Neiko2002/nsg
  // unsigned nTrees = (unsigned)0;
  // unsigned mLevel = (unsigned)0;
  // unsigned K = (unsigned)200;
  // unsigned L = (unsigned)200;
  // unsigned iter = (unsigned)10;
  // unsigned S = (unsigned)10;
  // unsigned R = (unsigned)100;



    // ------------------------ Deep1M ------------------------
  auto object_file      = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_base.fvecs)";
  auto query_file       = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_query.fvecs)";
  auto groundtruth_file = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_groundtruth.ivecs)";
  auto efanna_dir       = R"(e:/Data/Feature/Deep1M/efanna/)";

  size_t test_k = 100;
  size_t repeat_test = 1;
  std::vector<unsigned> L_search_parameter = { 100, 200, 300, 500, 800, 1200, 1600 };
    
  // Efanna
  unsigned nTrees = (unsigned)0;
  unsigned mLevel = (unsigned)0;
  unsigned K = (unsigned)50;
  unsigned L = (unsigned)70;
  unsigned iter = (unsigned)10;
  unsigned S = (unsigned)10;
  unsigned R = (unsigned)50;

  // // NSG efanna graph
  // unsigned nTrees = (unsigned)0;
  // unsigned mLevel = (unsigned)0;
  // unsigned K = (unsigned)200;
  // unsigned L = (unsigned)200;
  // unsigned iter = (unsigned)10;
  // unsigned S = (unsigned)10;
  // unsigned R = (unsigned)100;

  // SSG efanna graph 
  // unsigned nTrees = (unsigned)0;
  // unsigned mLevel = (unsigned)0;
  // unsigned K = (unsigned)200;
  // unsigned L = (unsigned)200;
  // unsigned iter = (unsigned)12;
  // unsigned S = (unsigned)10;
  // unsigned R = (unsigned)100;



  // ------------------------------------------------------
  // ------------------------ efanna ----------------------
  // ------------------------------------------------------
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb" << std::endl;

  std::cout << "Load Data" << std::endl;
  float* data_load = NULL;
  unsigned points_num, dim;
  load_fvecs(object_file, data_load, points_num, dim);
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading the data" << std::endl;

  std::cout << "Align Data" << std::endl;
  data_load = efanna2e::data_align(data_load, points_num, dim); // one must align the data before build
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after aligning the data" << std::endl;

  std::cout << "Create graph" << std::endl;
  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexGraph index(data_load, dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after creating the graph" << std::endl;


  // build graph
  auto efanna_file = string_format("%s/K%d_L%d_It%d_S%d_R%d_(nTrees%d_mLevel%d).efa", efanna_dir, K, L, iter, S, R, nTrees, mLevel);
  std::cout << "efanna_file: " << efanna_file << std::endl;
  if (exists_test(efanna_file) == false)
  {
    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", K);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("iter", iter);
    paras.Set<unsigned>("S", S);
    paras.Set<unsigned>("R", R);

    // use a kdtree to build the graph
    auto kdtree_file = string_format("%s/nTree%d_mLevel%d_K%d.kg", efanna_dir, nTrees, mLevel, K);
    if (nTrees > 0) 
    {

      // build kdtree
      if (exists_test(kdtree_file) == false)
      {
        efanna2e::Parameters kdtree_paras;
        kdtree_paras.Set<unsigned>("K", K);
        kdtree_paras.Set<unsigned>("nTrees", nTrees);
        kdtree_paras.Set<unsigned>("mLevel", mLevel);

        std::cout << "Create kdtree" << std::endl;
        efanna2e::IndexKDtree kdtree_index(dim, points_num, efanna2e::L2, nullptr);
        std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after creating the kdtree" << std::endl;

        std::cout << "Build kdtree" << std::endl;
        auto s = std::chrono::high_resolution_clock::now();
        kdtree_index.Build(points_num, data_load, kdtree_paras);
        auto e = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = e-s;
        std::cout <<"Time cost: "<< diff.count() << "\n";
        std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after building the kdtree" << std::endl;

        std::cout << "Store kdtree to " << kdtree_file << std::endl;
        kdtree_index.Save(kdtree_file.c_str());
      } 

      std::cout << "Load kdtree into graph" << std::endl;
      index.Load(kdtree_file.c_str());
      std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading kdtree into the graph" << std::endl;

      // refine kdtree
      std::cout << "Refine graph" << std::endl;
      auto s = std::chrono::high_resolution_clock::now();
      index.RefineGraph(data_load, paras);
      auto e = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = e-s;
      std::cout <<"Time cost: "<< diff.count() << "\n";
      std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after refining the graph" << std::endl;

    }
    else
    {
      // build without a kdtree
      std::cout << "Build graph" << std::endl;
      auto s = std::chrono::high_resolution_clock::now();
      index.Build(points_num, data_load, paras);
      auto e = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> elapsed_time = e-s;
      std::cout <<"Time cost: "<< elapsed_time.count() << "\n";
      std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after building the graph" << std::endl;
    }

    std::cout << "Store graph to " << efanna_file << std::endl;
    index.Save(efanna_file.c_str());
  }
  else
  {
    // load an existing graph from the drive
    std::cout << "Load graph from " << efanna_file << std::endl;
    index.Load(efanna_file.c_str());
    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading the graph" << std::endl;
  }

  // query data
  float* query_data = NULL;
  unsigned query_num, query_dim;
  load_fvecs(query_file, query_data, query_num, query_dim);
  query_data = efanna2e::data_align(query_data, query_num, query_dim); //one must align the data before build

  // query ground truth
  float* groundtruth_f = NULL;
  unsigned groundtruth_num, groundtruth_dim;
  load_fvecs(groundtruth_file, groundtruth_f, groundtruth_num, groundtruth_dim);
  const auto ground_truth = (uint32_t*)groundtruth_f; // not very clean, works as long as sizeof(int) == sizeof(float)
  const auto answers = get_ground_truth(ground_truth, groundtruth_num, groundtruth_dim, test_k);

  std::cout << "Test Top " << test_k << " for graph " << efanna_file << std::endl;
  auto ann = std::vector<unsigned>(test_k);

  // try differen L_search parameters
  for (float L_search : L_search_parameter)
  {
    // L search must be bigger or equal to k
    if(L_search < test_k)
      continue;

    efanna2e::Parameters query_paras;
    query_paras.Set<unsigned>("L_search", L_search);
    
    auto time_begin = std::chrono::steady_clock::now();
    size_t correct = 0;
    for (size_t t = 0; t < repeat_test; t++) {
      for (size_t i = 0; i < query_num; i++) {
        index.Search(query_data + i * query_dim, query_data, test_k, query_paras, ann.data());

        // compare answer with ann
        auto answer = answers[i];
        for (size_t r = 0; r < test_k; r++)
          if (answer.find(ann[r]) != answer.end()) 
            correct++;
      }
    }
    auto recall = 1.0f * correct / repeat_test / (query_num * test_k);

    auto time_end = std::chrono::steady_clock::now();
    auto time_us_per_query = (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) / (query_num * repeat_test);
    std::cout << string_format("L_search %5.f, recall %.4f, time_us_per_query %8d \n", L_search, recall, time_us_per_query);
    if (recall > 1.0)
      break;
  }

  // compute several stats
  // compute_stats(efanna_file.c_str(), dim, top_list_file);

  std::cout << "Finished" << std::endl;
  return 0;
}
