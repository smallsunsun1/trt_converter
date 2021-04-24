git submodule init
git submodule update

pushd third_party/async_lib
bash run.sh
popd

cd third_party/grpc
git submodule init
git submodule update
mkdir build && cd build 
cmake -DCMAKE_BUILD_TYPE=Release
