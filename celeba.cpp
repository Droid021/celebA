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
const char *kDataFolder = "./data";
// Periodic checkpoint
const int64_t kCheckpointEvery = 200;
// No of images to sample after checkpoint
const int64_t kSamplesPerCheckpoint = 10;
// Bool -> restore models and optimizers from prev saved checkpoints
const bool kRestoreFromCheckpoint = false;
// Log update with the loss value
const int64_t kLogInterval = 10;

using namespace torch;

struct GeneratorImpl : nn::Module
{
    GeneratorImpl(int kNoiseSize)
        : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256, 4)
                    .bias(false)),
          batch_norm1(256),
          conv2(nn::ConvTranspose2dOptions(256, 128, 3)
                    .stride(2)
                    .padding(1)
                    .bias(false)),
          batch_norm2(128),
          conv3(nn::ConvTranspose2dOptions(128, 64, 4)
                    .stride(2)
                    .padding(1)
                    .bias(false)),
          batch_norm3(64),
          conv4(nn::ConvTranspose2dOptions(64, 1, 4)
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
    torch::Tensor forward(torch::Tensor x)
    {
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

int main(int argc, const char *argv[])
{
    torch::manual_seed(42);

    //Device
    torch::Device device(torch::kCPU) if (torch::cuda::is_available())
    {
        std::cout << "Training on Cuda" << std::endl;
        device = torch::Device(torch::kCUDA);
    }
    Generator generator(kNoiseSize);
    generator->to(device)

        // Discriminator
        nn::Sequential discriminator(
            // layer 1
            nn::Conv2d(
                nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
            nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
            // layer 2
            nn::Conv2d(
                nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
            nn::BatchNorm2d(128),
            nn::LeakyRELU(nn::LeakyReLUOptions().negative_slope(0.2)),
            // layer 3
            nn::Conv2d(
                nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
            nn::BatchNorm2d(256),
            nn::LeakyRELU(nn::LeakyRELUOptions().negative_slope(0.2)),
            // layer 4
            nn::Conv2d(
                nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
            nn::Sigmoid());
    discriminator->to(device);

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
        torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

    torch::optim::Adam generator_optim(
        generator->parameters(),
        torch::optim::AdamOptions(2e-4););
    torch::optim::Adam discriminator_optim(
        discriminator->parameters(),
        torch::optim::AdamOptions(2e-4));

    // restore from checkpoint
    if (kRestoreFromCheckpoint)
    {
        torch::load(generator, "gen-checkpoint.pt");
        torch::load(generator_optim, "gen-optim.pt");
        torch::load(discriminator, "disc-checkpoint.pt");
        torch::load(discriminator_optim, "disc-optim.pt");
    }

    // Training loop
    int64_t checkpoint_counter = 1;
    for (int64_t epoch = 1; epoch <= kEpochs; ++epoch)
    {
        int64_t batch_index = 0;
        for (torch::data::Example<> &batch : *dataloader)
        {
            // Train discriminator with real images
            discriminator->zero_grad();
            torch::Tensor real_images = batch.data.to(device);
            torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
            torch::Tensor real_output = discriminator->forward(real_images);
            torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
            d_loss_real.backward();
            // Train with fake images
            torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
            torch::Tensor fake_images = generator->forward(noise);
            torch::Tensor fake_labels = torch::zeros(batch.target.size(0), device);
            torch::Tensor fake_output = discriminator->forward(fake_images.detach());
            torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
            d_loss_fake.backward();

            // Train generator
            generator->zero_grad();
            fake_labels.fill_(1);
            fake_output = discriminator->forward(fake_images);
            torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
            g_loss.backward();
            generator_optim.step();
            batch_index++;

            // log stuff
            if (batch_index % kLogInterval == 0)
            {
                std::printf(
                    "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f\n",
                    epoch,
                    kEpochs,
                    batch_index,
                    batches_per_epoch,
                    d_loss.item<float>(),
                    g_loss.item<float>());
            }

            //checkpoint model and other random stuff
            if (batch_index % kCheckpointEvery == 0)
            {
                torch::save(generator, "gen-checkpoint.pt");
                torch::save(generator_optim, "gen-optim.pt");
                torch::save(discriminator, "disc-checkpoint.pt");
                torch::save(discriminator_optim, "disc-optim.pt");

                // save sample generated images
                torch::Tensor samples = generator->forward(torch::randn({kSamplesPerCheckpoint, kNoiseSize, 1, 1}, device));
                torch::save((samples +1.0 )/2.0, 
                        torch::str("dcgan-", checkpoint_counter, ".pt"));
                std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
            }
        }
    }
    std::cout << "Training complete!" << std::endl;
};