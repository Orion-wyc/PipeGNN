/*  Version： 0.05
 *	Date: 2020.12.10
 *  Description: Assessing tensor-based NNs with MACC.
 *               Combining *AccPar and Pipelined training strategies ---> PipeG or HyPipe 
 *               hhh ~o(*￣▽￣*)ブ 
 */


#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <queue>
#include <map>
#include <set>
#include <cmath>
#include <time.h>
// #include "cnn.h"

#define MAX_NUM_LAYERS 1000
#define MAX_NUM_TYPES 3
#define MAX_NUM_DIMS 4
#define MAX_NUM_NODES 2
#define MAX_NUM_CONFIGS 10
#define ALPHA 0.6
#define INF 0x3f3f3f3f
// basic network and device configs

const int BATCH_SIZE = 128;
const double COMPUTATION_DENSITY = 13.7e12; // FLOPS - here 13.7 TFLOPS
const double INTRA_NODE_BANDWIDTH[MAX_NUM_NODES] = {10 * 1024 * 1024, 1 * 1024 * 1024}; // 10GB/s
const double INTER_NODE_BANDWIDTH = 1 * 1024 * 1024; //1GB/s

using namespace std;

// 前向声明
class Op ;
// 全局变量, 记录 NN 的层数
int global_layer_id = 0; 
Op* glidToOp[MAX_NUM_LAYERS];
// 暂时不用, 用于存储多分支
std::vector<std::vector<Op*> > parameters; 
// 暂时不使用
enum TYPE {TYPE1, TYPE2, TYPE3}; 


class Strategy {
public:
	Strategy() {}
	void AddConfig(int cur_type, int pre_type) {
		configs[cur_type].push_back(pre_type);
	}
	
	void PrintConfigs() {
		for (int i = 0; i < MAX_NUM_TYPES; i++) {
			printf("Strategy %d: [layer,type]...\n", i);
			for (int j = 0; j < global_layer_id; j++) { // 这里可以改成vector.size()
				printf("[%d,%d] ", j, configs[i][j]);
			}
			printf("\n");
		}
	}
private:
	vector<int> configs[MAX_NUM_TYPES];
};


class Tensor {
public:
	Tensor() : nDims(0), owner(NULL), idx(-1) {}

	Tensor(int _nDims, int* _dim, Op* _owner, int _idx)
		: nDims(_nDims), owner(_owner), idx(_idx) {
		for (int i = 0; i < nDims; i++) {
			dim[i] = _dim[i];
		}
	}

	Tensor(int _nDims, int dim0, int dim1, Op* _owner, int _idx)
		: nDims(_nDims), owner(_owner), idx(_idx) {
		assert(nDims == 2);
		dim[0] = dim0; dim[1] = dim1;
	}

	Tensor(int _nDims, int dim0, int dim1, int dim2, int dim3, Op* _owner, int _idx)
		: nDims(_nDims), owner(_owner), idx(_idx) {
		assert(nDims == 4);
		dim[0] = dim0; dim[1] = dim1; dim[2] = dim2; dim[3] = dim3;
	}

public:
	Op* owner;
	int idx, nDims, dim[MAX_NUM_DIMS];
// private:
};


class Op {
public:
	Op(string _name) : name(_name) {
		glid = global_layer_id ++;
		numInputs = 0;
		assert(glid < MAX_NUM_LAYERS);
		glidToOp[glid] = this; // 建立Op与layer全局glid的映射
	}

	void AddInputTensor(Tensor x) {
		numInputs ++;
		inputTensors.push_back(x);
		if (x.owner != NULL) {
			preOps.push_back(x.owner);
			x.owner->nextOps.push_back(this);
			x.owner->nextOpTensors.push_back(x);
		}
	}

	// virtual double compute(enum TYPE t) = 0;

	// virtual double communicate(enum TYPE ts, enmu TYPE td) = 0;

	virtual int getInputTensorSize() = 0;
	virtual int getOutputTensorSize() = 0;
	virtual double getComputeTime() = 0;
	virtual long getP() = 0; //单个元素所需要的浮点执行次数
	virtual int getW() = 0; // Op的权重大小
	virtual int getE() = 0; // Op的反向误差大小
	virtual int getF() = 0; // Op的输入张量大小
public:
	int glid, numInputs;
	std::string name;
	std::vector<Tensor> inputTensors, outputTensors, nextOpTensors;
	std::vector<Op*> nextOps, preOps;
};


class Conv2D : public Op {
public:
	Conv2D(int _outputSize, int _kernelH, int _kernelW, int _strideH, int _strideW,
	       int _paddingH, int _paddingW, Tensor x, std::string name)
		: outputSize(_outputSize), kernelH(_kernelH), kernelW(_kernelW), strideH(_strideH),
		  strideW(_strideW), paddingH(_paddingH), paddingW(_paddingW), Op(name) {
		assert(x.nDims == 4);
		batchSize = x.dim[0];
		inputSize = x.dim[1];
		inputHeight = x.dim[2];
		inputWidth = x.dim[3];
		outputHeight = 1 + (inputHeight + 2 * paddingH - kernelH) / strideH;

		outputWidth = 1 + (inputWidth + 2 * paddingW - kernelW) / strideW;
		assert(outputHeight > 0);
		assert(outputWidth > 0);
		AddInputTensor(x);
		Tensor y(4, batchSize, outputSize, outputHeight, outputWidth, this, 0);
		printf("Conv2D(%s):	input[%d %d %d %d] output[%d %d %d %d] kernel(%d %d) stride(%d %d) padding(%d %d)\n",
		       name.c_str(), batchSize, inputSize, inputHeight, inputWidth,
		       batchSize, outputSize, outputHeight, outputWidth, kernelH, kernelW,
		       strideH, strideW, paddingH, paddingW);
		outputTensors.push_back(y);
		// computeTime = measure_conv2d_time(batchSize, inputSize, inputHeight, inputWidth, 
        //                                    outputSize, outputHeight, outputWidth,
        //                                    kernelH, kernelW, strideH, strideW,
        //                                    paddingH, paddingW);
	}

	int getInputTensorSize() {
		return batchSize * inputSize * inputWidth * inputHeight;
	}

	int getOutputTensorSize() {
		return batchSize * outputSize * outputWidth * outputHeight;
	}

	double getComputeTime(){
		return computeTime;
	}

	long getP(){
		long P1 = this->getOutputTensorSize() * (inputSize * kernelH * kernelW + inputSize * kernelH * kernelW - 1);
		long P2 = batchSize * inputSize * kernelH * kernelW * outputHeight\
		          * outputWidth * (outputSize + outputSize - 1);
		long P3 = 0;
		return P1 + P2 + P3;
	}
	int getW(){
		return  kernelH * kernelW * inputSize * outputSize;
	}
	int getE(){
		return batchSize * inputSize * inputWidth * inputHeight;
	}
	int getF(){
		return batchSize * inputSize * inputWidth * inputHeight;
	}
private:
	int batchSize, inputSize, outputSize, inputWidth, inputHeight, outputWidth, outputHeight;
	int kernelH, kernelW, strideH, strideW, paddingH, paddingW;
	double computeTime;
};


class Pool2D : public Op {
public:
	Pool2D(int _kernelH, int _kernelW, int _strideH, int _strideW,
	       int _paddingH, int _paddingW, Tensor x, std::string name)
		: kernelH(_kernelH), kernelW(_kernelW), strideH(_strideH), strideW(_strideW),
		  paddingH(_paddingH), paddingW(_paddingW), Op(name) {
		assert(x.nDims == 4);
		batchSize = x.dim[0];
		outputSize = x.dim[1];
		inputHeight = x.dim[2];
		inputWidth = x.dim[3];
		outputHeight = 1 + (inputHeight + 2 * paddingH - kernelH) / strideH;
		outputWidth = 1 + (inputWidth + 2 * paddingW - kernelW) / strideW;
		assert(outputHeight > 0);
		assert(outputWidth > 0);
		AddInputTensor(x);
		Tensor y(4, batchSize, outputSize, outputHeight, outputWidth, this, 0);
		outputTensors.push_back(y);
		printf("Pool2D(%s): input[%d %d %d %d] output[%d %d %d %d]\n", name.c_str(), x.dim[0], \
				x.dim[1], x.dim[2], x.dim[3],  y.dim[0], y.dim[1], y.dim[2], y.dim[3]);
		// computeTime = measure_pool2d_time(batchSize, outputSize, inputHeight, inputWidth,
        //                                    outputHeight, outputWidth,
        //                                    kernelH, kernelW, strideH, strideW,
        //                                    paddingH, paddingW);
	}

	int getInputTensorSize() {
		return batchSize * inputSize * inputWidth * inputHeight;
	}

	int getOutputTensorSize() {
		return batchSize * outputSize * outputWidth * outputHeight;
	}

	double getComputeTime(){
		return computeTime;
	}

	long getP(){
		return 0;
	}
	int getW(){
		return 0;
	}
	int getE(){
		return 0;
	}
	int getF(){
		return 0;
	}
private:
	int batchSize, inputSize, outputSize, inputWidth, inputHeight, outputWidth, outputHeight;
	int kernelH, kernelW, strideH, strideW, paddingH, paddingW;
	double computeTime;
};

class Flat : public Op {
public:
	Flat(Tensor x, std::string name) : Op(name) {
		assert(x.nDims == 4);
		outputSize = x.dim[1] * x.dim[2] * x.dim[3];
		AddInputTensor(x);
		Tensor y(2, x.dim[0], outputSize, this, 0);
		outputTensors.push_back(y);
		printf("Flat(%s): input[%d %d %d %d] output[%d %d]\n", name.c_str(), x.dim[0], \
				x.dim[1], x.dim[2], x.dim[3],  y.dim[0], y.dim[1]);
	}

	int getInputTensorSize() { // flatten 输入输出尺寸不改变
		return outputSize;
	}

	int getOutputTensorSize() {
		return outputSize;
	}

	double getComputeTime(){
		return computeTime;
	}

	long getP(){
		return 0;
	}
	int getW(){
		return 0;
	}
	int getE(){
		return 0;
	}
	int getF(){
		return 0;
	}
private:
	int outputSize;
	double computeTime;
};


class Softmax : public Op {
public:
	Softmax(int _batchSize, int _inputSize, int _outputSize, bool softmax, bool lstm_linear, Tensor x, std::string name)
		: batchSize(_batchSize), inputSize(_inputSize), outputSize(_outputSize), Op(name) {
		assert(x.nDims == 2);
		assert(x.dim[0] == batchSize);
		assert(x.dim[1] == inputSize);
		AddInputTensor(x);
		assert(numInputs == 1);
		Tensor y(2, batchSize, outputSize, this, 0);
		outputTensors.push_back(y);
		printf("Linear(%s): input[%d %d] output[%d %d]\n", name.c_str(), batchSize, inputSize, batchSize, outputSize);
	}
	int getInputTensorSize() { // flatten 输入输出尺寸不改变
		return batchSize * inputSize;
	}

	int getOutputTensorSize() {
		return batchSize * outputSize;
	}

	double getComputeTime(){
		return computeTime;
	}
	long getP(){ // FC的计算量咋算呀:这里修改函数功能，变成计算整个算子的计算量
		long P1 = batchSize * outputSize * (inputSize + inputSize - 1); //fwd
		long P2 = inputSize * outputSize * (batchSize + batchSize - 1); //bwd \partial W
		long P3 = batchSize * inputSize * (outputSize + outputSize - 1); // bwd \partial X
		return P1 + P2 + P3;
	}
	int getW(){
		return inputSize * outputSize;
	}
	int getE(){
		return batchSize * outputSize;
	}
	int getF(){
		return batchSize * outputSize;
	}
private:
	int batchSize, inputSize, outputSize;
	double computeTime;
};


Tensor add_conv_layer(Tensor t, int outputSize, int kernelX, int kernelY,
                      int strideX, int strideY, int paddingX, int paddingY,
                      std::string name)
{
  Conv2D* conv = new Conv2D(outputSize, kernelX, kernelY, strideX, strideY,
                            paddingX, paddingY, t, name);
  return conv->outputTensors[0];
}

Tensor add_pool_layer(Tensor t, int kernelX, int kernelY, int strideX, int strideY,
                      int paddingX, int paddingY, std::string name)
{
  Pool2D* pool = new Pool2D(kernelX, kernelY, strideX, strideY, paddingX, paddingY,
                            t, name);
  return pool->outputTensors[0];
}

Tensor add_flat_layer(Tensor t, std::string name)
{
  Flat* flat = new Flat(t, name);
  assert(flat->outputTensors.size() == 1);
  return flat->outputTensors[0];
}

Tensor add_linear_layer(Tensor t, int outputSize, bool softmaxLayer, std::string name)
{
  Softmax* softmax = new Softmax(t.dim[0], t.dim[1], outputSize, softmaxLayer, false/*linear*/, t, name);
  assert(softmax->outputTensors.size() == 1);
  return softmax->outputTensors[0];
}

void build_alexnet_model()
{
	// init_cudnn();
	printf("Network Structure: \n");
	Tensor x(4, BATCH_SIZE, 3, 224, 224, NULL, 0);
	Tensor t = add_conv_layer(x, 64, 11, 11, 4, 4, 2, 2, "conv1");
	t = add_pool_layer(t, 3, 3, 2, 2, 0, 0, "pool2");
	t = add_conv_layer(t, 192, 5, 5, 1, 1, 2, 2, "conv3");
	t = add_pool_layer(t, 3, 3, 2, 2, 0, 0, "pool4");
	t = add_conv_layer(t, 384, 3, 3, 1, 1, 1, 1, "conv5");
	t = add_conv_layer(t, 256, 3, 3, 1, 1, 1, 1, "conv6");
	t = add_conv_layer(t, 256, 3, 3, 1, 1, 1, 1, "conv7");
	t = add_pool_layer(t, 3, 3, 2, 2, 0, 0, "pool8");
	t = add_flat_layer(t, "flat");
	t = add_linear_layer(t, 4096, false, "linear9");
	t = add_linear_layer(t, 4096, false, "linear10");
	t = add_linear_layer(t, 1000, true, "linear11");
	std::vector<Op*> opList;
	for (int i = 0; i < global_layer_id; i++) {
		opList.clear();
		opList.push_back(glidToOp[i]);
		parameters.push_back(opList);
	}
}

//先写一手模拟的过程,带宽与计算能力随意设置
double ComputeTime(int l, int type){
	Op* op = glidToOp[l];
	int P = op->getP();
	double Ecp = ALPHA * P / COMPUTATION_DENSITY;
	return Ecp;
}

double InterCommunicationTime(int l, int tt, int t){
	double Ecm = 0;
	assert(t>=0 && t<=2 && tt>=0 && tt<=2);
	Op* op_l = glidToOp[l];
	if((tt==0 && t==0) || (tt==1 && t==2) || (tt==2 && t==1)){
		Ecm = 0;
	}else if((tt==0 && t==1) || (tt==2 && t==0)){
		Ecm = ALPHA * (1-ALPHA) * (op_l->getE() + op_l->getF()) / INTER_NODE_BANDWIDTH;
	}else{
		Ecm = (1-ALPHA) * (op_l->getE()) / INTER_NODE_BANDWIDTH;
	}

	return Ecm;
}

double IntraCommunicationTime(int l, int t){
	double Ecm = 0;
	assert(t>=0 && t<=2);
	Op* op = glidToOp[l];
	if(t==0){
		Ecm = op->getW() / INTRA_NODE_BANDWIDTH[0];
	} else if(t==1){
		Ecm = op->getF() / INTRA_NODE_BANDWIDTH[0];
	}else{
		Ecm = op->getE() / INTRA_NODE_BANDWIDTH[0];
	}
	return Ecm;
}

int main() {
	// 构造网络, 目前在考虑是否需要测 pooling 的开销, 论文中是未考虑的
	build_alexnet_model();

	double dp[MAX_NUM_LAYERS][MAX_NUM_TYPES];  // 存储执行总时间

	Strategy strategy;

	for (int i = 0; i < MAX_NUM_LAYERS; i++) {
		for (int j = 0; j < MAX_NUM_TYPES; j++) {
			dp[i][j] = INF;
		}
	}
	// 首层结果初始化 dp[0][0],dp[0][1],dp[0][2]
	dp[0][0] = ComputeTime(0, 0) ;
	dp[0][1] = ComputeTime(0, 1) ;
	dp[0][2] = ComputeTime(0, 2) ;

	for (int l = 1; l < global_layer_id; l++) {
		for (int t = 0; t < MAX_NUM_TYPES; t++) {
			int type;
			double Ecp = ComputeTime(l, t);
			double Ecm1 = IntraCommunicationTime(l, t);
			for (int tt = 0; tt < MAX_NUM_TYPES; tt++) {
				double Ecm2 = InterCommunicationTime(l, tt, t);
				double Ecm = Ecm1 + Ecm2;
				if (dp[l - 1][tt] + Ecp + Ecm < dp[l][t]) {
					dp[l][t] = dp[l - 1][tt] + Ecp + Ecm;
					type = tt;
				}
			}
			strategy.AddConfig(t, type);
		}
	}
	// 添加最后一层的
	for (int t = 0; t < MAX_NUM_TYPES; t++) {
		strategy.AddConfig(t, t);
		cout << dp[global_layer_id - 1][t] << endl;
	}
	strategy.PrintConfigs();

	return 0;
}