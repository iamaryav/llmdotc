#include<stdio.h>
#include<stdlib.h>



// cpu code
// x = x + causal_attn(ln(x))
// two input array and need to add them
//
//
//

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, const int N){
	for(int i = 0; i < N; i++){
		out[i] = inp1[i] + inp2[i];
	}
}


// gpu code




