import subprocess
import os

def compile_cmakhov_lib_gcc():
    """
    compiling with gcc (requires gsl):
    $ gcc -Wall -fPIC -c makhov_library.c
    $ gcc -Wall -lgsl -lgslcblas -shared -o liblimpid.so library.o
    """

    CWD = os.getcwd()
    os.chdir(CWD + "/src/limpid/c_libraries")

    print("C library for makhov was not found!")
    print("Compiling makhov_library.c!")
    cmd_0 = ["gcc", "-Wall", "-fPIC", "-c", "makhov_library.c"]
    cmd_1 = ["gcc", "-Wall", "-lgsl", "-lgslcblas", "-shared", "-o", "makhov_library.so", "makhov_library.o"]
    p = subprocess.Popen(cmd_0)
    p.wait()
    p = subprocess.Popen(cmd_1)
    p.wait()
    os.remove("makhov_library.o")
    print("Compiling completed!")

    os.chdir(CWD)