#include <fstream>
#include <iostream>

#include <vector>
#include <unordered_set>
#include <filesystem>
#include <limits>


static void load_fvecs(const char* filename, float*& data, unsigned& num,unsigned& dim){
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

template<typename... Args>
std::string string_format(const char* fmt, Args... args)
{
    size_t size = snprintf(nullptr, 0, fmt, args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt, args...);
    return buf;
}

static inline bool exists_test(const std::string &name)
{
    auto f = std::ifstream(name.c_str());
    return f.good();
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


static auto read_top_list(const char* fname, size_t& d_out, size_t& n_out)
{
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{})
    {
        std::cerr << "error when accessing top list file" << fname << " size is: " << file_size << " message: " << ec.message() << std::endl;
        perror("");
        abort();
    }

    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open())
    {
        std::cerr << "could not open " << fname << std::endl;
        perror("");
        abort();
    }

    uint32_t dims;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(int));
    assert((dims > 0 && dims < 1000000) || !"unreasonable dimension");
    assert(file_size % ((dims + 1) * 4) == 0 || !"weird file size");
    size_t n = file_size / ((dims + 1) * 4);

    d_out = dims;
    n_out = n;

    auto x = std::make_unique<uint32_t[]>(n * (dims + 1));
    ifstream.seekg(0);
    ifstream.read(reinterpret_cast<char*>(x.get()), n * (dims + 1) * sizeof(uint32_t));
    if (!ifstream) 
        assert(ifstream.gcount() == static_cast<int>(n * (dims + 1)) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) 
        memmove(&x[i * dims], &x[1 + i * (dims + 1)], dims * sizeof(uint32_t));

    ifstream.close();
    return x;
}


static uint32_t compute_reachablity_count(std::vector<std::vector<uint32_t>>& graph) {

    auto graph_size = graph.size();
    uint32_t reachable_count = 0;

    unsigned L = 100; // L_search
    unsigned seed = 1998;
    std::mt19937 rng(seed);
    std::vector<unsigned> init_ids(L);
    efanna2e::GenRandom(rng, init_ids.data(), L, graph_size);
    
    // flood fill from this entrance position
    auto checked_ids = std::vector<bool>(graph_size);
    auto check = std::vector<uint32_t>();

    // start with the first nodes
    for (unsigned s: init_ids) {
        checked_ids[s] = true;
        check.emplace_back(s);
    }
    
    // repeat as long as we have nodes to check
	while(check.size() > 0) {	

        // neighbors which will be checked next round
        auto check_next = std::vector<uint32_t>();

        // get the neighbors to check next
        for (auto &&check_index : check) {
 
            auto& neighbor_indizies = graph[check_index];        
            auto const &neighbors = graph[check_index];

            for (int i = 0; i < neighbor_indizies.size(); i++) {
                auto neighbor_index = neighbor_indizies[i];
                
                if(checked_ids[neighbor_index] == false) {
                    checked_ids[neighbor_index] = true;
                    check_next.emplace_back(neighbor_index);
                }
            }
        }

        check = std::move(check_next);
    }

    // how many nodes have been checked
    uint32_t checked_node_count = 0;
    for (size_t i = 0; i < graph_size; i++)
        if(checked_ids[i])
            checked_node_count++;

    std::cout << "Seed Reachablity " << checked_node_count << " of " << graph_size << " vertices" << std::endl;
    return checked_node_count;
}

static double compute_avg_reach(std::vector<std::vector<uint32_t>>& graph, std::vector<bool>& is_source_vertex) {
    auto graph_size = graph.size();

    // remember those vertices which have a very high reach
    uint32_t best_vertex_checked_count = 0;                             // current highest reach count
    auto best_vertices_checked_ids = std::vector<std::vector<bool>>();  // which vertices can be reach by one of the best vertices
    auto best_reach_indices = std::vector<uint32_t>(graph_size);        // a list of all vertices containing the index to one of the best vertices
    std::fill(best_reach_indices.begin(), best_reach_indices.end(), graph_size);

    // numbers of vertices to be reached starting from each vertex
    auto reach_counts = std::vector<uint32_t>(graph_size);
    uint64_t avg_reach = 0;
    for (size_t entry_id = 0; entry_id < graph_size; entry_id++)
    {
        // flood fill from this entrance position and try to reach the target_id
        auto checked_ids = std::vector<bool>(graph_size);
        auto check = std::vector<uint32_t>();
        auto check_next = std::vector<uint32_t>();

        // start with the first node
        checked_ids[entry_id] = true;
        check.emplace_back(entry_id);
        
        // we try to speed up the process by reaching a vertex which can reach all other vertices
        // or we find a vertex which is currently the best vertex and copy its reach
        bool reach_all = false;
        bool reach_best = false;        

        // repeat as long as we have nodes to check
        auto check_ptr = &check;
        auto check_next_ptr = &check_next;
		while(check_ptr->size() > 0 && reach_all == false) {	

            // neighbors which will be checked next round
            check_next_ptr->clear();

            // get the neighbors to check next
            for (size_t c = 0; c < check_ptr->size() && reach_all == false; c++) {
                const auto check_index = check_ptr->at(c);
                const auto& neighbor_indizies = graph[check_index];        

                if(neighbor_indizies.size() == 0)
                    std::cout << "zero out-degree for vertex " << check_index << std::endl;

                for (int i = 0; i < neighbor_indizies.size(); i++) {
                    auto neighbor_index = (uint32_t)neighbor_indizies[i];
                    
                    if(checked_ids[neighbor_index] == false) {
                        checked_ids[neighbor_index] = true;
                        check_next_ptr->emplace_back(neighbor_index);

                        // found a vertex which can reach all other vertices
                        if(reach_counts[neighbor_index] == graph_size) {
                            reach_all = true;
                            break;
                        }

                        // found one of the best vertices or a vertex which can reach the best -> copy the reach of the best
                        if(reach_best == false && reach_all == false) {
                            auto best_reach_index = best_reach_indices[neighbor_index];
                            if(best_reach_index < graph_size) {
                                best_reach_indices[entry_id] = best_reach_index;

                                auto& best_vertex_checked_ids = best_vertices_checked_ids[best_reach_index];
                                for (size_t b = 0; b < graph_size; b++) 
                                    checked_ids[b] = checked_ids[b] | best_vertex_checked_ids[b];
                                reach_best = true;
                            }
                        }
                    }
                }
            }

            auto buffer = check_ptr;
            check_ptr = check_next_ptr;
            check_next_ptr = buffer;
        }

        // how many nodes have been checked
        uint32_t reach_count = reach_all ? graph_size : 0;
        if(reach_all == false)
            for (size_t i = 0; i < graph_size; i++)
                reach_count += checked_ids[i];
        reach_counts[entry_id] = reach_count;
        avg_reach += reach_count;
        
        // is this a new best vertex?
        if(is_source_vertex[entry_id] == false && (reach_count > best_vertex_checked_count || reach_best == false)) {
            best_vertex_checked_count = reach_count;
            best_reach_indices[entry_id] = best_vertices_checked_ids.size();

            if(reach_all)
                std::fill(checked_ids.begin(), checked_ids.end(), true);
            best_vertices_checked_ids.emplace_back(checked_ids);
            //std::cout << "Current best vertex " << entry_id << " at index " << best_reach_indices[entry_id] << " has a reach of " << reach_count << " vertices and is a source node " << is_source_vertex[entry_id] << std::endl;
        }

        if((entry_id+1) % 10000 == 0)
            std::printf("Avg reach is %.2f after checking %zd of %zd vertices\n", ((double)avg_reach)/(entry_id+1), (entry_id+1), graph_size);
    }  

    std::printf("Avg reach is %.2f after checking %zd of %zd vertices\n", ((double)avg_reach)/graph_size, graph_size, graph_size);
    return ((double)avg_reach)/graph_size;
}

static void compute_stats(const char* graph_file, const uint32_t feature_dims, const char* top_list_file) {
    std::cout << "Compute graph stats of " << graph_file << std::endl;


    size_t top_list_dims;
    size_t top_list_count;
    const auto all_top_list = read_top_list(top_list_file, top_list_dims, top_list_count);
    std::cout << "Load TopList from file" << top_list_file << " with " << top_list_count << " elements and k=" << top_list_dims << std::endl;

    auto index = efanna2e::IndexGraph(feature_dims, top_list_count, efanna2e::L2, nullptr);
    index.Load(graph_file);
    auto graph = index.getCompactGraph();
    auto graph_size = graph.size();

    
    // compute the graph quality
    uint64_t perfect_neighbor_count = 0;
    uint64_t total_neighbor_count = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto& neighbor_indizies = graph[n];
        auto edges_per_node = neighbor_indizies.size();

        // get top list of this node
        auto top_list = all_top_list.get() + n * top_list_dims;
        if(top_list_dims < edges_per_node) {
            std::cerr << "TopList for " << n << " is not long enough has " << edges_per_node << " elements has " << top_list_dims << std::endl;
            edges_per_node = (uint16_t) top_list_dims;
        }
        total_neighbor_count += edges_per_node;

        // check if every neighbor is from the perfect neighborhood
        for (uint32_t e = 0; e < edges_per_node; e++) {
            auto neighbor_index = neighbor_indizies[e];

            // find in the neighbor ini the first few elements of the top list
            for (uint32_t i = 0; i < edges_per_node; i++) {
                if(neighbor_index == top_list[i]) {
                    perfect_neighbor_count++;
                    break;
                }
            }
        }
    }
    auto perfect_neighbor_ratio = (float) perfect_neighbor_count / total_neighbor_count;
    auto avg_edge_count = (float) total_neighbor_count / graph_size;

    // compute the min, and max out degree
    uint16_t min_out =  std::numeric_limits<uint16_t>::max();
    uint16_t max_out = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto& neighbor_indizies = graph[n];
        auto edges_per_node = neighbor_indizies.size();

        if(edges_per_node < min_out)
            min_out = edges_per_node;
        if(max_out < edges_per_node)
            max_out = edges_per_node;
    }

    // compute the min, and max in degree
    auto in_degree_count = std::vector<uint32_t>(graph_size);
    for (uint32_t n = 0; n < graph_size; n++) {
        auto& neighbor_indizies = graph[n];
        auto edges_per_node = neighbor_indizies.size();

        for (uint32_t e = 0; e < edges_per_node; e++) {
            auto neighbor_index = neighbor_indizies[e];
            in_degree_count[neighbor_index]++;
        }
    }

    uint32_t min_in = std::numeric_limits<uint32_t>::max();
    uint32_t max_in = 0;
    uint32_t source_nodes = 0;
    auto is_source_vertex = std::vector<bool>(graph_size);
    std::fill(is_source_vertex.begin(), is_source_vertex.end(), false);
    for (uint32_t n = 0; n < graph_size; n++) {
        auto in_degree = in_degree_count[n];

        if(in_degree < min_in)
            min_in = in_degree;
        if(max_in < in_degree)
            max_in = in_degree;

        if(in_degree == 0) {
            is_source_vertex[n] = true;
            source_nodes++;
            //std::cout << "Node " << n << " has zero incoming connections" << std::endl;
        }
    }

    auto reachability_count = compute_reachablity_count(graph);
    auto avg_reach = compute_avg_reach(graph, is_source_vertex);

    std::printf("GQ %.4f, avg degree %.1f, min_out %d, max_out %d, min_in %d, max_in %d, source nodes %d, search reachability count %d, exploration avg reach %.2f, node count %zd\n", perfect_neighbor_ratio, avg_edge_count, min_out, max_out, min_in, max_in, source_nodes, reachability_count, avg_reach, graph_size);
}
