#ifndef MY_RESNET_H
#define MY_RESNET_H

#include <vector>
#include <string.h> // void *memset(void *str, int c, size_t n)

#include "common.h"

#define DTYPE_MIN -1e6
#define PADDING_VALUE 0.0

/** TO BE OPTIMIZED
 * change to inplace: BatchNorm2d, ReLU, operator+
 * improve efficiency: Conv2d
 **/


class InferenceBase
{ /* abstract base class for inference modules */
public:
    InferenceBase(): inplace(false) {}
    explicit InferenceBase(bool is_inplace): inplace(is_inplace) {}
    virtual ~InferenceBase() {}
    
    virtual TypedTensor forward(const TypedTensor& x) =0;
    virtual void forward_inplace(TypedTensor& x) const { }
    virtual void load_from(ifstream &in_file) =0;
    
    virtual bool get_inplace(void) const { return inplace; }
    virtual void set_inplace(bool new_inplace) { inplace = new_inplace; }

protected:
    bool inplace;
};


void readline_from_ifstream(ifstream& in_file);


class AdaptiveAveragePool2d;

class Conv2d: public InferenceBase
{
public:
    Conv2d(
        int in_channels, int out_channels, vector<int> kernel_size, 
        int padding = 0, int stride = 1, bool use_bias = false);
    Conv2d(const Conv2d& obj);
    Conv2d(Conv2d && obj);
    // Conv2d(ifstream& in_file);
    ~Conv2d();
    TypedTensor forward(const TypedTensor& x);
    const vector<int> get_kernel_size(void) const;
    void load_weight_from(ifstream& in_file) { weight.load_from(in_file);}
    void load_weight_from(const char *filename) { weight.load_from(filename); }
    void load_bias_from(ifstream& in_file) { if(muse_bias) bias.load_from(in_file); }
    void load_bias_from(const char *filename) { if(muse_bias) bias.load_from(filename); }
    void load_from(ifstream& in_file);
    // void load_from(const char *filename);

private:
    TypedTensor weight;
    TypedTensor bias;
    int mpadding;
    int mstride;
    bool muse_bias;

friend class AdaptiveAveragePool2d;
};


TypedTensor unfoldTensor(const TypedTensor& x, const vector<int>& kernel_size, int padding, int stride);


class MaxPool2d: public InferenceBase
{
public:
    MaxPool2d(int kernel_size, int stride = -1, int padding = 0);
    MaxPool2d(const MaxPool2d& obj);
    MaxPool2d(MaxPool2d && obj);
    ~MaxPool2d();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    int mkernel_size;
    int mstride;
    int mpadding;
};


class BatchNorm2d: public InferenceBase
{
public:
    BatchNorm2d(int num_features, double eps = 1e-5, bool is_inplace = true);
    BatchNorm2d(const BatchNorm2d& obj);
    BatchNorm2d(BatchNorm2d && obj);
    ~BatchNorm2d();
    TypedTensor forward(const TypedTensor& x);
    void forward_inplace(TypedTensor& x) const;
    void load_weight_from(ifstream& in_file) { weight.load_from(in_file); }
    void load_weight_from(const char *filename) { weight.load_from(filename); }
    void load_bias_from(ifstream& in_file) { bias.load_from(in_file); }
    void load_bias_from(const char *filename) { bias.load_from(filename); }
    void load_mean_from(ifstream& in_file) { running_mean.load_from(in_file); }
    void load_mean_from(const char *filename) { running_mean.load_from(filename); }
    void load_var_from(ifstream& in_file) { running_var.load_from(in_file); }
    void load_var_from(const char *filename) { running_var.load_from(filename); }
    void load_from(ifstream& in_file);

private:
    TypedTensor weight;
    TypedTensor bias;
    TypedTensor running_mean;
    TypedTensor running_var;
    TypedTensor processed_var;
    int mnum_features;
    double meps;
};


class Linear: public InferenceBase
{
public:
    Linear(int in_features, int out_features, bool bias = true);
    Linear(const Linear& obj);
    Linear(Linear && obj);
    ~Linear();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    TypedTensor weight;
    TypedTensor mbias;
};


class ReLU: public InferenceBase
{
public:
    ReLU(bool is_inplace = true);
    // ReLU(const ReLU& obj);
    // ReLU(ReLU &&obj);
    ~ReLU();
    TypedTensor forward(const TypedTensor& x);
    void forward_inplace(TypedTensor& x) const;
    void load_from(ifstream& in_file);

private:

};


class Tanh: public InferenceBase
{
public:
    Tanh(bool is_inplace = true);
    ~Tanh();
    TypedTensor forward(const TypedTensor& x);
    void forward_inplace(TypedTensor& x) const;
    void load_from(ifstream& in_file);
};


class Flatten: public InferenceBase
{
public:
    Flatten(bool is_inplace = true);
    ~Flatten();
    TypedTensor forward(const TypedTensor& x);
    void forward_inplace(TypedTensor& x) const;
    void load_from(ifstream& in_file) {}
};



class BasicBlock: public InferenceBase
{
public:
    BasicBlock(int in_channels, int out_channels);
    BasicBlock(const BasicBlock& obj);
    BasicBlock(BasicBlock && obj);
    ~BasicBlock();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    Conv2d conv1;
    BatchNorm2d bn1;
    ReLU relu1;
    Conv2d conv2;
    BatchNorm2d bn2;
    Conv2d shortcut;
    ReLU relu2;
    bool use_shortcut;
};


// pytorch-compatible: for ResNet34-v2
class BasicBlock_v2: public InferenceBase
{
public:
    BasicBlock_v2(int in_channels, int out_channels);
    BasicBlock_v2(const BasicBlock_v2& obj);
    BasicBlock_v2(BasicBlock_v2 && obj);
    ~BasicBlock_v2();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    Conv2d conv1;
    BatchNorm2d bn1;
    ReLU relu1;
    Conv2d conv2;
    BatchNorm2d bn2;
    Conv2d shortcut;
    BatchNorm2d bn_s; // like pytorch
    ReLU relu2;
    bool use_shortcut;
};


class DuplicateBasicBlocks: public InferenceBase
{
public:
    DuplicateBasicBlocks(int n_repeat, int in_channels, int out_channels);
    ~DuplicateBasicBlocks();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    vector<BasicBlock> basic_blocks;
};


// pytorch-compatible
class DuplicateBasicBlocks_v2: public InferenceBase
{
public:
    DuplicateBasicBlocks_v2(int n_repeat, int in_channels, int out_channels);
    ~DuplicateBasicBlocks_v2();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    vector<BasicBlock_v2> basic_blocks;
};


// 3-layer basicblock for resnet50+
class Bottleneck: public InferenceBase
{
public:
    // `in_channels` is the input channels, `filter1_channels` is the output channels of the first 1x1filter
    // the output channels for the whole bottleneck is 4 x filter1_channels 
    Bottleneck(int in_channels, int filter1_channels, int stride3x3 = 1, bool use_shortcut_ = false); 
    Bottleneck(const Bottleneck& obj);
    Bottleneck(Bottleneck&& obj);
    ~Bottleneck();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    Conv2d conv1;
    BatchNorm2d bn1;
    ReLU relu1;
    Conv2d conv2;
    BatchNorm2d bn2;
    ReLU relu2;
    Conv2d conv3;
    BatchNorm2d bn3;
    Conv2d shortcut;
    BatchNorm2d bn_s; // for shortcut
    ReLU relu3;
    bool use_shortcut;
};


class DuplicateBottlenecks: public InferenceBase
{
public:
    DuplicateBottlenecks(int n_repeat, int in_channels, int filter1_channels, bool first_bottleneck = false);
    ~DuplicateBottlenecks();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    vector<Bottleneck> bottle_necks;
};



class AvgPool2d: public InferenceBase
{
public:
    AvgPool2d(
        const vector<int>& kernel_size_, 
        const vector<int>& stride_ = {}, 
        int padding_ = 0);
    ~AvgPool2d();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    vector<int> kernel_size;
    vector<int> stride;
    vector<int> padding;
};


class GlobalAveragePool2d_flatten: public InferenceBase
{
public:
    GlobalAveragePool2d_flatten();
    ~GlobalAveragePool2d_flatten();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
};


class AdaptiveAveragePool2d_flatten: public InferenceBase
{
public:
    AdaptiveAveragePool2d_flatten(const vector<int>& _output_size);
    ~AdaptiveAveragePool2d_flatten();
    TypedTensor forward(const TypedTensor& x); // not a const function, since it modifies `pooling_kernel`
    void load_from(ifstream& in_file);

private:
    vector<int> output_size;
    // Conv2d *pooling_kernel;
};


class Softmax: public InferenceBase
{
public:
    Softmax();
    ~Softmax();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
};


// NEURAL NETWORK ARCHITECTURES

// class ResNet34: public InferenceBase
// {
// public:
//     ResNet34(int init_channel = 64);
//     ~ResNet34();
//     TypedTensor forward(const TypedTensor& x);
//     void load_from(ifstream& in_file);

// private:
//     vector<InferenceBase *> network;
// };

class LeNet5: public InferenceBase
{
public:
    LeNet5(int num_classes);
    ~LeNet5();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    vector<InferenceBase *> network;
};


// pytorch-compatible: shortcut uses an additional batchnorm layer
class ResNet34_v2: public InferenceBase
{
public:
    ResNet34_v2(int num_classes);
    ~ResNet34_v2();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    vector<InferenceBase *> network;
};


class ResNet50: public InferenceBase
{
public:
    ResNet50(int num_classes);
    ~ResNet50();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    vector<InferenceBase *> network;
};


class VGG16: public InferenceBase
{
public:
    VGG16(int num_classes);
    ~VGG16();
    TypedTensor forward(const TypedTensor& x);
    void load_from(ifstream& in_file);

private:
    vector<InferenceBase *> network;
};


#endif