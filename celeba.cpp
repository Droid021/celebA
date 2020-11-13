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
    nn::ConvTranspose2d conv1, conv2, conv3, conv4;
    nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3, batch_norm4;

};

TORCH_MODULE(Generator);

int main(int argc, const char* argv[]){
    torch::manual_seed(42);
    
    //Device
    torch::Device device(torch::kCPU)
    if (torch::cuda::is_available()) {
        std::cout << "Training on Cuda" << std::endl;
        device = torch::Device(torch::kCUDA);
    }
    Generator generator(kNoiseSize);
    generator -> to(device)

    // Discriminator
    nn::Sequential discriminator(
        // layer 1
        nn::Conv2d(
            nn::Conv2dOptions(1,64,4).stride(2).padding(1).bias(false)
        ),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // layer 2
        nn::Conv2d(
            nn::Conv2dOptions(64,128,4).stride(2).padding(1).bias(false)
        ),
        nn::BatchNorm2d(128),
        nn::LeakyRELU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // layer 3
        nn::Conv2d(
            nn::Conv2dOptions(128,256,4).stride(2).padding(1).bias(false)
        ),
        nn::BatchNorm2d(256),
        nn::LeakyRELU(nn::LeakyRELUOptions().negative_slope(0.2)),
        // layer 4
        nn::Conv2d(
            nn::Conv2dOptions(256, 1,3).stride(1).padding(0).bias(false)
        ),
        nn::Sigmoid()
    );
    discriminator -> to(device);

    std::vector<double> norm_mean = {0.485, 0.456, 0.406};
    std::vector<double> norm_std = {0.229, 0.224, 0.225};
    // Load Data
    auto dataset = torch::data::datasets::CELEBA(kDataFolder)
                    .map(torch::data::transforms::Normalize<>(norm_mean, norm_std))
                    .map(torch::data::transforms::Stack<>());
    const int64_t = batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));
    // Dataloader
    auto dataloader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2)
    );

    
};