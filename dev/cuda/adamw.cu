


# adamw optimizer

void adamw_cpu(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_params, float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0) {
    // list of all the params
    // calculate the m_t, v_t, bias correction, weight decay
    // update 
    // adamw
    //
    // for each params we have to calculate the adamw
    for (int i = 0; i < num_params; i++){
        float param = params_memory[i];
        float grad = grads_memory[i];

    }


}

int main(int argc, char** argv){

    return 0;
}
