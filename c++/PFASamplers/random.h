#pragma once
#include <iostream>
#include <random>
#include <omp.h>

class MTGen {
public:
    MTGen(int num_threads  = 1, int seed = 374): _num_threads(num_threads), _gens() {
      std::random_device rd;
        auto seed_gen = std::mt19937_64();
        seed_gen.seed(seed);
        std::uniform_int_distribution<int> seed_distribution(1, 100000000);
        for (int i = 0; i < num_threads ; i++) {
            int seed = seed_distribution(seed_gen);
            //std::cout << "Seed for thread "<< i << ": "<< seed << std::endl;
            auto gen = std::mt19937_64(seed);
            _gens.push_back(gen);
        }
    };
    std::mt19937_64& get() { return _gens[omp_get_thread_num()]; };
    std::mt19937_64& get(int thread) { return _gens[thread]; };
    int num_threads() {return _num_threads;};
    
private:
    int _num_threads;
    std::vector<std::mt19937_64> _gens;
};