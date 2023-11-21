#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <chrono>
#include <algorithm>

using namespace std::chrono;
using namespace std;

vector<int> generate_random_vector(int number_of_elements, int value_of_max_element){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, value_of_max_element);
    vector<int> vec;

    for (int i = 0; i < number_of_elements; ++i) {
        vec.push_back(dis(gen));
    }
    return vec;
}

void merge(std::vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (i = left; i <= right; i++) {
        arr[i] = temp[i - left];
    }
}

void mergeSortOpenMP(vector<int>& vec, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        #pragma omp parallel sections default(none) shared(vec, left, mid, right)
        {
            #pragma omp section
            {
                mergeSortOpenMP(vec, left, mid);
            }

            #pragma omp section
            {
                mergeSortOpenMP(vec, mid + 1, right);
            }
        }

        merge(vec, left, mid, right);
    }
}

int main() {

    int number_of_runs = 10;
    int number_of_threads = 1;
    int number_of_elements = 10000000;

    omp_set_dynamic(0);
    omp_set_num_threads(number_of_threads);

    vector<double> runs;

    cout << "Number of threads: " << number_of_threads << endl;
    cout << "Number of elements: " << number_of_elements << endl;
    for (int i = 0; i < number_of_runs; ++i) {
        vector<int> vec = generate_random_vector(number_of_elements, 10000001);
//        cout << "Sample of the vector: ";
//        for (int i = 0; i < 10; ++i) {
//            cout << vec[i] << " ";
//        }
//        cout << endl;

        auto start = high_resolution_clock::now();
        mergeSortOpenMP(vec, 0, number_of_elements - 1);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);
        double time = duration.count() / 1000000.0;
        runs.push_back(time);
    }

    double final_time = 0;

    for (int i = 0; i < number_of_runs; ++i) {
        final_time += runs[i];
        cout << "Run " << i + 1 << " time: " << runs[i] << endl;
    }
    cout << endl << "Mean time: " << final_time/number_of_runs << endl;
    vector<double> sorted_runs = runs;
    sort(sorted_runs.begin(), sorted_runs.end());
    cout << "Uncertainty: " << (sorted_runs[sorted_runs.size() - 1] - sorted_runs[0])/2 << endl;
    return 0;
}