#include <torch/torch.h>

#include <cmath>
#include <cstdio>
#include <iostream>

// Noise vector for the generator
const int64_t kNoiseSize = 100;
// Batch size
const int64_t kBatchSize = 64;
// Epochs
const int64_t kEpochs = 30;
// Mnist data store
const char* kDataFolder = "./data";
// Periodic checkpoint 
const int64_t kCheckpointEvery = 200;
// No of images to sample after checkpoint
const int64_t kSamplesPerCheckpoint = 10;
// Bool -> restore models and optimizers from prev saved checkpoints
const bool kRestoreFromCheckpoint = false;
// Log update with the loss value
const int64_t kLogInterval = 10;

using namespace torch;

struct GeneratorImpl : nn::Module {
    GeneratorImpl(int kNoiseSize)
        :   conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256,4)
                    .bias(false)),
            batch_norm1(256),
            conv2(nn::ConvTranspose2dOptions(256, 128,3)
                    .stride(2)
                    .padding(1)
                    .bias(false)),
            batch_norm2(128),
            conv3(nn::ConvTranspose2dOptions(128, 64, 4)
                    .stride(2)
                    .padding(1)
                    .bias(false)),
            batch_norm3(64),
            conv4(nn::ConvTranspose2dOptions(64,1,4)
                    .stride(2)
                    .padding(1)
                    .bias(false))
    // register modules
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
    }

    // forward
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::tanh(conv4(x));
        return x;
    }

}