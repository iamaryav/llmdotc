#include<math.h>

// adamw optimizer

void adamw_cpu(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_params, float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0) {

    // calculate the m_t, v_t, bias correction, weight decay
    // for each params we have to calculate the adamw
    // by how much we should nudge the weight and which direction
    for (int i = 0; i < num_params; i++){
        float param = params_memory[i];
        float grad = grads_memory[i];
        // first moment (momentum)
        float m = beta1 * m_memory[i] + (1.0f - beta1) * grad;
        // second moment (RMSprop)
        float v = beta2 * v_memory[i] + (1.0f - beta2) * grad * grad;

        // bias correct both moments
        float m_hat = m / (1.0f - powf(beta1, t);
        float v_hat = v / (1.0f - powf(beta2, t);

        //updates
        m_memory[i] = m;
        v_memory[i] = v;
        params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);

    }

}

float* random_float(size_t N){
    float* temp = (float*)malloc(N * sizeof(float));
    for(int i = 0; i < N; i++){
        temp[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0 // range 0..1
    }

}

int main(int argc, char** argv){

    // running adamw cpu
    int B = 2;
    int T = 4;
    int C = 8;
    int N = B * T * C; // number of input tokens in a batch

    float* params_memory;
    float* grads_memory;
    float* m_memory;
    float* v_memory;
    int t;
    long num_params;
    float learning_rate;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;

    // count of params
    // assigne the memory
    // numbers of params for now its upto us
    params_memory = random_float01(num_params);
    grads_memory = random_float01(num_params);
    m_memory = (float*)malloc(num_params * sizeof(float));
    v_memory = (float*)malloc(num_params * sizeof(float));
    // they all needs to initialize with random values

    





    return 0;
}
