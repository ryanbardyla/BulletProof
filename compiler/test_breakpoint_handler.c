#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

void sigtrap_handler(int sig) {
    printf("\n🔴 BREAKPOINT HIT! (SIGTRAP received)\n");
    printf("   This proves the INT3 instruction was executed!\n");
    exit(0);  // Exit cleanly after catching breakpoint
}

int main() {
    // Install SIGTRAP handler
    signal(SIGTRAP, sigtrap_handler);
    
    // Fork and exec the NeuronLang program
    pid_t pid = fork();
    if (pid == 0) {
        // Child: trace itself and run the program
        printf("🚀 Running NeuronLang program with breakpoint detection...\n\n");
        execl("./test_breakpoint_simple", "test_breakpoint_simple", NULL);
        perror("execl");
        exit(1);
    } else {
        // Parent: wait for child
        int status;
        wait(&status);
        if (WIFSIGNALED(status) && WTERMSIG(status) == SIGTRAP) {
            printf("\n✅ Breakpoint support verified!\n");
        }
    }
    return 0;
}