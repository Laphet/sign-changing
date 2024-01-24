This is a code repository for using CEM-GMsFEM to solve so-called sign-changing problems.

We also improved our previous codes in some sense, now the codes look much cleaner!

According to my tests, it is strongly advisable to install **scikit-umfpack** instead of the original sparse solver (actually the sequential version of **SuperLU**) provided by **scipy**.