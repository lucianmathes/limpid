import pkg_resources
import subprocess
import os

LIB_FOLDER = os.path.expanduser("~/.limpid")
C_PATH = pkg_resources.resource_filename(__name__, "makhov.c")

def compile_cmakhov_lib_gcc():
    """
    compiling with gcc (requires gsl):
    $ gcc -Wall -fPIC -c makhov_library.c
    $ gcc -Wall -lgsl -lgslcblas -shared -o liblimpid.so library.o
    """

    CWD = os.getcwd()
    if not os.path.exists(LIB_FOLDER):
        os.mkdir(LIB_FOLDER)
    os.chdir(LIB_FOLDER)

    print("C library for makhov was not found!")
    print("Compiling makhov_library.c!")
    cmd_0 = ["gcc", "-Wall", "-fPIC", "-c", C_PATH]
    cmd_1 = ["gcc", "-Wall", "-lgsl", "-lgslcblas", "-shared", "-o", "makhov.so", "makhov.o"]
    p = subprocess.Popen(cmd_0)
    p.wait()
    p = subprocess.Popen(cmd_1)
    p.wait()
    os.remove("makhov.o")
    print("Compiling completed!")

    os.chdir(CWD)
