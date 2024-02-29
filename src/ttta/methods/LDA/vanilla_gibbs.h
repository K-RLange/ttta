#include <math.h>
#include <time.h>
#include <stdlib.h>

void vanilla_gibbs(unsigned long long * w_vec, unsigned int * as_vec, unsigned long long * d_vec, unsigned long long * vk_mat,  unsigned long long * dk_mat,
                   unsigned long long * v_sum, double * alpha, double * gamma, long double * probs,
                   int K, long V, long D, long W, int iter, int start){
    unsigned int k0, k1;
    unsigned long long vv=0, dd=0, ww, ii;
    long double random;
    srand(time(NULL));

    for (ww = 0; ww < W; ww++) {
        vv = w_vec[ww];
        dd=d_vec[ww];
        k0 = as_vec[ww];
        as_vec[ww] = k0;
        vk_mat[vv*K+k0]++;
        dk_mat[dd*K+k0]++;
        v_sum[k0]++;
    }

    int sum_as_same;
    for (ii = 0; ii < iter; ii++) {
        sum_as_same=0;
        for (int ww = start; ww < W; ww++) {
            vv = w_vec[ww];
            dd=d_vec[ww];
            k0 = as_vec[ww];
            k1 = k0;
            vk_mat[vv*K+k0]--;
            dk_mat[dd*K+k0]--;
            v_sum[k0]--;
            probs[0] = (long double) (vk_mat[vv*K] + gamma[0]) * (dk_mat[dd*K] + alpha[0]) / (fmax(v_sum[0], 1) + V * gamma[0]);

            for (int kk = 1; kk < K; kk++) {
                probs[kk] =(long double) (vk_mat[vv*K+kk] + gamma[kk]) * (dk_mat[dd*K+kk] + alpha[kk]) / (fmax(v_sum[kk], 1) + V * gamma[kk]) + probs[kk -1];
            }
            if (probs[K-1]==0) {
              printf("probs are equal to %f\n", probs[K-1]);
              for (int newk = 0; newk < K; ++newk) probs[newk]=1.0/K;
            }
            // print the probs:
//            for (int kk = 0; kk < K; kk++) {
//                printf("%f ", probs[kk]);
//            }

            random = (long double) rand() / (RAND_MAX) * probs[K-1];
//            printf("%f\nprobs:", random);
            for (int kk = 0; kk < K; kk++) {
                if (probs[kk] > random) {
                    k1 = kk;
                    break;
                }
            }

            as_vec[ww] = k1;
            vk_mat[vv*K+k1]++;
            dk_mat[dd*K+k1]++;
            v_sum[k1]++;
            if (k0 == k1) {sum_as_same++;}

        }
//        printf("%s", "Iteration ");
//        printf("%d", ii+1);
//        printf("%s", " of ");
//        printf("%d", iter);
//        printf("%s", " Share of stable assignments: ");
//        printf("%f ", (double) sum_as_same/W);
//        printf("%c", '\n');
    }
}



void load_wk_mat(unsigned long long * w_vec, unsigned int * as_vec, unsigned long long * vk_mat, int K, long V, long W){
    unsigned int k0;
    unsigned long long vv=0, ww;
    srand(time(NULL));
    for (ww = 0; ww < W; ww++) {
        vv = w_vec[ww];
        k0 = as_vec[ww];
        as_vec[ww] = k0;
        vk_mat[vv*K+k0]++;
    }
}

void load_dk_mat(unsigned long long * d_vec, unsigned int * as_vec, unsigned long long * dk_mat, int K, long V, long D){
    unsigned int k0;
    unsigned long long vv=0, dd=0;
    srand(time(NULL));

    for (dd = 0; dd < D; dd++) {
        vv = d_vec[dd];
        k0 = as_vec[dd];
        as_vec[dd] = k0;
        dk_mat[vv*K+k0]++;
    }
}


void final_assigment(unsigned long long * w_vec, unsigned int * as_vec, unsigned long long * d_vec, unsigned long long * vk_mat,  unsigned long long * dk_mat,
                     unsigned long long * v_sum, double * alpha, double * gamma, double * probs, int K, long V,long D, long W,
                     int iter, int start){
    double max_prob;
    unsigned int k0, k1;
    unsigned long long vv=0, dd=0, ww;
    for (int ww = start; ww < W; ww++) {
        vv = w_vec[ww];
        dd=d_vec[ww];
        k0 = as_vec[ww];

        probs[0] = (double) (vk_mat[vv*K] + gamma[0]) * (dk_mat[dd*K] + alpha[0]) / v_sum[0] ;
        for (int kk = 1; kk < K; kk++) {
             probs[kk] =(double) (vk_mat[vv*K+kk] + gamma[kk]) * (dk_mat[dd*K+kk] + alpha[kk]) / v_sum[kk];
        }

        max_prob = probs[0];
        k1 = 0;

        for (int kk = 1; kk < K; kk++) {
            if (probs[kk] > max_prob) {
                k1 = kk;
                max_prob = probs[kk];
            }
        }
        as_vec[ww] = k1;
    }
}