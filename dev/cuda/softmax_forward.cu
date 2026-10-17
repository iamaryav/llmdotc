


//---------------------------------------------------------------------------------------
// CPU code ref

void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C
        float* out_row = out + i * C

        // calculate max for each row
        for (int j = 0; j < 
    }

}


//---------------------------------------------------------------------------------------
