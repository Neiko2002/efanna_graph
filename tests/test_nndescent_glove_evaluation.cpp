//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>

#include <vector>
#include <unordered_set>


void load_data(const char* filename, float*& data, unsigned& num,unsigned& dim){// load data with sift10K pattern
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

static std::vector<std::unordered_set<uint32_t>> get_ground_truth(const uint32_t* ground_truth, const size_t ground_truth_size, const uint32_t ground_truth_dims, const size_t k)
{
    auto answers = std::vector<std::unordered_set<uint32_t>>(ground_truth_size);
    answers.reserve(ground_truth_size);
    for (int i = 0; i < ground_truth_size; i++)
    {
        auto& gt = answers[i];
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) 
            gt.insert(ground_truth[ground_truth_dims * i + j]);
    }

    return answers;
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


  auto object_file      = R"(e:/Data/Feature/GloVe/glove-100/glove-100_base.fvecs)";
  auto query_file       = R"(e:/Data/Feature/GloVe/glove-100/glove-100_query.fvecs)";
  auto groundtruth_file = R"(e:/Data/Feature/GloVe/glove-100/glove-100_groundtruth.ivecs)";
  auto efanna_file      = R"(e:/Data/Feature/GloVe/efanna/glove-100_K300_L320_It12_S10_R200.efa)";

  size_t k = 100;

  // database features 
  std::cout << "Load basedata and graph" << std::endl;
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(object_file, data_load, points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim); // one must align the data before build

  // final efanna graph
  auto index = efanna2e::IndexGraph(data_load, dim, points_num, efanna2e::L2, nullptr);
  index.Load(efanna_file);
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading base data and graph" << std::endl;

  // query data
  float* query_data = NULL;
  unsigned query_num, query_dim;
  load_data(query_file, query_data, query_num, query_dim);
  query_data = efanna2e::data_align(query_data, query_num, query_dim);//one must align the data before build

  // query ground truth
  float* groundtruth_f = NULL;
  unsigned groundtruth_num, groundtruth_dim;
  load_data(groundtruth_file, groundtruth_f, groundtruth_num, groundtruth_dim);
  const auto ground_truth = (uint32_t*)groundtruth_f; // not very clean, works as long as sizeof(int) == sizeof(float)
  const auto answers = get_ground_truth(ground_truth, groundtruth_num, groundtruth_dim, k);

  std::cout << "Test graph " << efanna_file << std::endl;
  auto ann = std::vector<unsigned>(k);

  // try differen L_search parameters
  std::vector<unsigned> L_search_parameter = { 300, 800, 1000, 2000, 4000, 8000, 16000, 32000 };
  for (float L_search : L_search_parameter)
  {
    // L search must be bigger or equal to k
    if(L_search < k)
      continue;

    efanna2e::Parameters query_paras;
    query_paras.Set<unsigned>("L_search", L_search);
    
    auto time_begin = std::chrono::steady_clock::now();
    
    size_t correct = 0;
    for (size_t i = 0; i < query_num; i++) {
      index.Search(query_data + i * query_dim, query_data, k, query_paras, ann.data());

      // compare answer with ann
      auto answer = answers[i];
      for (size_t r = 0; r < k; r++)
        if (answer.find(ann[r]) != answer.end()) correct++;
    }

    auto time_end = std::chrono::steady_clock::now();
    auto time_us_per_query = (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) / query_num;
    auto recall = 1.0f * correct / (query_num * k);
    std::cout << "L_search " << L_search << ", recall " << recall << ", time_us_per_query " << time_us_per_query << std::endl;
    if (recall > 1.0)
      break;
  }

  return 0;
}
