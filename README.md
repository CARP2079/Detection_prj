# Detection_prj
目前只公开了项目的测试部分，生成部分代码和脚本后续公开

该项目目前基于ZCU102和ZCU104开发板开发实现，需要下载的附属文件可通过此[百度云链接](https://pan.baidu.com/s/1BOMHiFKS7u8wBrJhunRI1A)下载,提取码：n8w6 

下载后使用系统烧写工具Win32DiskImager或者balenaEtcher将其烧录到SD卡上，然后将此工程下开发板对应版本的文件目录，拷贝到SD卡的/home/root目录下。

文件目录介绍
```
Detection_prj
├── README.md
├── LICENSE
├── SDcard_dir/home/root         # 拷贝到sd卡中文件系统目录下
│        ├── dpu_sw_optimize     # xilinx官方的处理硬件问题的脚本
│        ├── kernel_test         # 对硬件中hls核单独做测试用的
│        ├── mynet               # 目标检测网络
│        ├── testcolor           # 测试预处理核用的纯色图片
│        └── testpic             # 目标检测网络的测试图像目录，可从百度云中拷贝进1000多张图进此目录 
├── net_for_ZCU104               # 存放网络硬件编译的运行指令的动态链接库，仅适用于ZCU104镜像
│        └── libdpumodeltestnet.so
└── libtest_uhwapi.so            # OpenCL交叉编译的适用于调度hls核的动态链接库
```

# 复现本项目的操作流程
- 下载对应开发板的镜像，使用系统烧写工具Win32DiskImager或者balenaEtcher将其烧录到SD卡上。
- 将此工程下SDcard_dir/home/root文件目录下的文件，拷贝到SD卡的/home/root目录下。
- 拷贝 libdpumodeltestnet.so 到 mynet目录下，解压并拷贝百度云中的testpic中的图片到testpic目录，拷贝 libtest_uhwapi.so 到SD卡的/usr/lib目录下，拷贝百度云中的vitis-ai_v1.2_dnndk.tar.gz 到/home/root目录。
- 将SD卡插入FPGA开发板，开机，通过串口输入 ifconfig 获得局域网ip地址。
- PC linux上终端输入如下（XX为ip地址），远程控制开发板操作系统
```
% export DISPLAY=:0.0 
% ssh -X root@XX.XX.XX.XX
```
- 输入如下命令安装dnndk V1.2运行环境
```sh
tar -xzvf vitis-ai_v1.2_dnndk.tar.gz
cd vitis-ai_v1.2_dnndk
./install.sh
```
- 执行以下命令来启动xrt和调整Xilinx ZCU104板上运行DPU型号的过流故障极限避免出错。（每次重启均需输入）
```sh
./dpu_sw_optimize/zynqmp/zynqmp_dpu_optimize.sh
source /mnt/sd-mmcblk0p1/init.sh
```
- 更改目录到 mynet 下，执行以下命令来搬移xclbin文件。如需要运行 kernel_test 目录下的程序也需要在该目录如此操作。
```sh
cp /media/sd-mmcblk0p1/dpu.xclbin ./
```
- 运行测试程序
```sh
python3 custnet_platform_v10.py
```


