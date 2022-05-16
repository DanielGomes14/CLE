#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[])
{
    int rank, size;
    char msg[] = "How are you guys?";
    char response[] = "Good and you?";
    char* recvData;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    
    if(size == 1)
    {
        printf("There is no one to send messages to. Exiting...");
        return EXIT_SUCCESS;
    }

    if(rank == 0)   // first process
    {
        for(int i = 1; i < size; i++)
        {
            printf("Sending message to everyone from rank <%d>\n", rank);
            MPI_Send(msg, strlen(msg), MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }

        for(int i = 1; i < size; i++)
        {
            recvData = malloc(100);
            for(int j = 0; j < 100; j++)
                recvData[j] = '\0';

            MPI_Recv(recvData, 100, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Received message from rank <%d>: %s\n", i, recvData);
            
            free(recvData);
        }
    }
    else    // other processes
    {
        recvData = malloc(100);
        for(int j = 0; j < 100; j++)
            recvData[j] = '\0';

        MPI_Recv(recvData, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received message at rank <%d> from rank <0>: %s\n", rank, recvData);

        printf("Sending response to rank <0> from rank <%d>\n", rank);
        MPI_Send(response, strlen(response), MPI_CHAR, 0, 0, MPI_COMM_WORLD);

        free(recvData);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;

}