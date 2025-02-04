

# NOTE: do the following with sudo before run this script
# - sudo apt update
# - sudo apt install libomp-dev

# NOTE: if you are using venv, make sure to activate it before running this script
# for conda, you might also neeed to add env activation to the .bashrc file

# NOTE: cd to the repository folder if you prefer

# clone the acados repository
git clone https://github.com/acados/acados.git

# setup the env variable ACADOS_DIR
cd acados
ACADOS_DIR="$PWD"
echo "ACADOS_DIR: $ACADOS_DIR"
export ACADOS_ROOT_FOLDER=$ACADOS_DIR

# update submodule
git submodule update --recursive --init

# make and build
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OPENMP=ON ..
make install -j4

# install to pipy
pip install -e $ACADOS_ROOT_FOLDER/interfaces/acados_template
# export env variables
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$ACADOS_ROOT_FOLDER/lib"" >> ~/.bashrc
echo "export ACADOS_SOURCE_DIR="$ACADOS_ROOT_FOLDER"" >> ~/.bashrc
source ~/.bashrc

# install t_renderer automatically when running the examples
cd $ACADOS_ROOT_FOLDER/examples/acados_python/getting_started/
python3 minimal_example_ocp.py <<!
y
!
# the last line will run the example and automatically 
# input a 'y' to install t_renderer
