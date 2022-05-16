#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[])
{
    int rank, size;
    char data[] = "I am here!";
    char* recvData;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    

    if(rank == 0)   // first process
    {
            if(size == 1)   // only 1 process
            {
                printf("Transmitted message from rank <%d>: %s\n", rank, data);
                MPI_Send(data, strlen(data), MPI_CHAR, 0, 0, MPI_COMM_WORLD);

                recvData = malloc(100);
                for(int j = 0; j < 100; j++)
                    recvData[j] = '\0';
                MPI_Recv(recvData, strlen(data), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("Received message at rank <%d>: %s\n", rank, recvData);

            }
            else    // more then 1 process
            {
                printf("Transmitted message from rank <%d>: %s\n", rank, data);
                MPI_Send(data, strlen(data), MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD);

                recvData = malloc(100);
                for(int j = 0; j < 100; j++)
                    recvData[j] = '\0';
                MPI_Recv(recvData, strlen(data), MPI_CHAR, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("Received message at rank <%d>: %s\n", rank, recvData);
            }
    }
    // else if(rank == size - 1)   // last process
    // {
    //     recvData = malloc(100);
    //     for(int j = 0; j < 100; j++)
    //         recvData[j] = '\0';
    //     MPI_Recv(recvData, strlen(data), MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     printf("Received message at rank <%d>: %s\n", rank, recvData);

    //     printf("Transmitted message from rank <%d>: %s\n", rank, data);
    //     MPI_Send(data, strlen(data), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    // }
    else    // other processes
    {
        recvData = malloc(100);
        for(int j = 0; j < 100; j++)
            recvData[j] = '\0';
        MPI_Recv(recvData, strlen(data), MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received message at rank <%d>: %s\n", rank, recvData);

        if(rank == size - 1)
        {
            printf("Transmitted message from rank <%d>: %s\n", rank, data);
            MPI_Send(data, strlen(data), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
        else{
            printf("Transmitted message from rank <%d>: %s\n", rank, data);
            MPI_Send(data, strlen(data), MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD);
        }

        // printf("Transmitted message from rank <%d>: %s\n", rank, data);
        // MPI_Send(data, strlen(data), MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}