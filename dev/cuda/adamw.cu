// adamw optimizer

#include<stdio.h>
#include<stdlib.h>

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
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        //updates
        m_memory[i] = m;
        v_memory[i] = v;
        params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);

    }

}

float* random_float(size_t N){
    float* temp = (float*)malloc(N * sizeof(float));
    for(int i = 0; i < N; i++){
        temp[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range 0..1
    }
    return temp;
}


float* random_float01(size_t N){
    float* temp = (float*)malloc(N * sizeof(float));
    for(int i = 0; i < N; i++){
        temp[i] = ((float)rand() / RAND_MAX); // range 0..1
    }
    return temp;
}

int main(int argc, char** argv){


    float* params_memory;
    float* grads_memory;
    float* m_memory;
    float* v_memory;

    // intializing the hyper params
    const int t = 5;
    const long num_params = 1024;
    const float learning_rate = 1e-3;
    const float beta1 = 0.9;
    const float beta2 = 0.999;
    float eps = 1e-8;
    float weight_decay = 0.0;

    // initializing these with random values
    params_memory = random_float(num_params);
    grads_memory = random_float(num_params);
    m_memory = random_float(num_params);
    v_memory = random_float01(num_params);

    // before calculation 
    for(int i = 0; i < 5; i++){
        printf("params_memory: %8.4f grads_memory: %8.4f, m_memory: %8.4f, v_memory: %8.4f\n", params_memory[i], grads_memory[i], m_memory[i], v_memory[i]);
    }
    adamw_cpu(params_memory, grads_memory, m_memory, v_memory, t, num_params, learning_rate, beta1, beta2, eps, weight_decay);

    // after calculation
    printf("After adamw calculation\n");
    for(int i = 0; i < 5; i++){
        printf("params_memory: %8.4f grads_memory: %8.4f, m_memory: %8.4f, v_memory: %8.4f\n", params_memory[i], grads_memory[i], m_memory[i], v_memory[i]);
    }
    
    free(params_memory);
    free(grads_memory);
    free(m_memory);
    free(v_memory);

    return 0;
}
