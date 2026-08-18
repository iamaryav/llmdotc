



void crossentroy_forward_cpu(float* losses, const float* probs, int* target, int B, int T, int V){
    // -log(probs)
    // P_i is flattened array

    // B * T * V for each batch (skips over B size)
    // T * V for each token inside a batch (skips over t tokens)
    // B * T * v + t * v points to the start of the probablity vector of the t_th token
    // for target
    // B * T + t
    // simply, math of pointing to the right probablity vector and right target index
}
