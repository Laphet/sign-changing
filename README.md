This is a code repository for using CEM-GMsFEM to solve so-called sign-changing problems.

I also improved my previous codes in some sense, now the codes look much cleaner!

According to my tests, it is strongly advisable to install **scikit-umfpack** instead of the original sparse solver (actually the sequential version of **SuperLU**) provided by **scipy**.

If you are lucky to have Intel CPUs, you are highly recommended to use pypardiso, which grants you the mighty onemkl-pardiso solver (parallel in thread-wise)!