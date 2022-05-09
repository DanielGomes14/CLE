#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/**
 * Write a program that determines wich are the numbers with the highest
 * and lowest values among those of a random generated sequence. 
 * 
 * cada processo gera uma sequencia
 * cada process faz reduce pra min e max
 * enviar resultados para root(gather do root???)
 * root faz reduce pro min e max dos resultados obtidos
 * print dos resultados finais
 */

/*
    4 processos total

    root cria sequencia 16 valores
    scatter separa valores em 4 arrays com 4 valores cada
    reduce do array de 4 valores pra obter min/max
    scatter de 1 valor
    reduce 1 valor pra min/max

    [16]
    scatter da sequencia
    4 [4]
    reduce(min/max) de cada sub-sequencia
    4 [1]
    scatter de cada sub-sequencia
    4 1
    reduce(min/max) de cada sub-sequencia
    1 min/max

*/

int main(int argc, char** argv)
{

    int rank, size, min, max;
    int* sendData;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    // calculate random values
    if(rank == 0)
    {
        srandom(getpid());
        sendData = malloc(16 * sizeof(int));
        for(int i = 0; i < 16; i++)
        {
            sendData[i] = ((double) rand() / RAND_MAX) * 1000;
            printf("%d\t", sendData[i]);
        }
        printf("\n");
    }

    // scatter the sequence
    int* recvData;
    recvData = malloc(4 * sizeof(int));
    MPI_Scatter(sendData, 4, MPI_INT, recvData, 4, MPI_INT, 0, MPI_COMM_WORLD);

    // for(int i = 0; i < size; i++) {
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     if (i == rank) {
    //         printf("<%d> %d %d %d %d\n", i, recvData[0], recvData[1], recvData[2], recvData[3]);
    //     }
    // }

    // first reduce(get mins and maxs of each sub-sequence)
    int *mins, *maxs;
    mins = malloc(4 * sizeof(int));
    maxs = malloc(4 * sizeof(int));
    MPI_Reduce(recvData, mins, 4, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(recvData, maxs, 4, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // only the root process obtains the results
    // for(int i = 0; i < size; i++) {
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     if (i == rank) {
    //         printf("<%d> %d %d %d %d\n", i, mins[0], mins[1], mins[2], mins[3]);
    //     }
    // }

    // scatter sub-sequence
    MPI_Scatter(mins, 1, MPI_INT, &min, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(maxs, 1, MPI_INT, &max, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // for(int i = 0; i < size; i++)
    //     if(rank == i)
    //         printf("<%d> %d %d\n", rank, min, max);

    // second reduce(get min and max)
    int final_min, final_max;
    MPI_Reduce(&min, &final_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&max, &final_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0)
        printf("Min: <%d>\tMax: <%d>\n", final_min, final_max);

    MPI_Finalize();
    
    return EXIT_SUCCESS;
}