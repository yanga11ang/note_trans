mkdir release
g++ -m64 ./base/Base.cpp -fPIC -shared -o ./release/Base.dll -pthread -O3 -march=native

#D:\py_workspace\machine_learning\experiment\paper\OpenKE\openke
