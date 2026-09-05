double seconds = milliseconds / 1000.0;

double gb = bytes / 1e9;

double bandwidth = gb / seconds;

std::vector<float> pageable(N);

float* pinned = nullptr;

cudaMallocHost(&pinned, bytes);

auto start = std::chrono::steady_clock::now();

for (int i = 0; i < REPEAT; ++i) {

    cudaMemcpy(
        d_data,
        pageable.data(),
        bytes,
        cudaMemcpyHostToDevice
    );
}

auto stop = std::chrono::steady_clock::now();

for (int i = 0; i < REPEAT; ++i) {

    cudaMemcpy(
        d_A,
        pinned,
        bytes,
        cudaMemcpyHostToDevice
    );
}
