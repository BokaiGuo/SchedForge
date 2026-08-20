void schedforge_neon_matmul(const float*, const float*, float*);

extern "C" void* memcpy(void* destination, const void* source, unsigned long size) {
    auto* output = static_cast<unsigned char*>(destination);
    const auto* input = static_cast<const unsigned char*>(source);
    for (unsigned long index = 0; index < size; ++index) output[index] = input[index];
    return destination;
}

extern "C" void* memset(void* destination, int value, unsigned long size) {
    auto* output = static_cast<unsigned char*>(destination);
    for (unsigned long index = 0; index < size; ++index)
        output[index] = static_cast<unsigned char>(value);
    return destination;
}

extern "C" void _start() {
    const float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8};
    const float weights[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float output[16] = {};
    schedforge_neon_matmul(input, weights, output);
    const unsigned long status =
        output[0] == 1.0F && output[5] == 6.0F && output[10] == 3.0F && output[15] == 8.0F
        ? 0 : 1;
    asm volatile("mov x0, %0\nmov x8, #93\nsvc #0" :: "r"(status) : "x0", "x8");
    __builtin_unreachable();
}
