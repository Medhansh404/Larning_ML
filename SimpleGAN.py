{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNUvp5WNhYPYYJMECjCeQCR",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Medhansh404/Learning_ML/blob/main/SimpleGAN.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "s-9KOyGIkzn4"
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torchvision\n",
        "import torchvision.datasets as datasets\n",
        "import torch.utils.data\n",
        "import torchvision.transforms as transforms\n",
        "from torch.utils.tensorboard import SummaryWriter"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class Discriminator(nn.Module):\n",
        "  def __init__ (self, in_dim):\n",
        "    super().__init__()\n",
        "    self.disc = nn.Sequential(\n",
        "        nn.Linear(in_dim, 128),\n",
        "        nn.LeakyReLU(0.1),\n",
        "        nn.Linear(128, 1),\n",
        "        nn.Sigmoid()\n",
        "    )\n",
        "    def forward(self, x):\n",
        "      return self.disc(x)"
      ],
      "metadata": {
        "id": "GkOEqe7xlihR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class Generator(nn.Module):\n",
        "  def __init__(self, n_dim, in_dim):\n",
        "    super().__init__()\n",
        "    self.gen = nn.Sequential(\n",
        "        nn.Linear(n_dim, 256),\n",
        "        nn.LeakyReLU(0.1),\n",
        "        nn.Linear(256, in_dim),\n",
        "        nn.Tanh()\n",
        "    )\n",
        "\n",
        "    def forward(self, x):\n",
        "      return self.gen(x)\n"
      ],
      "metadata": {
        "id": "KAEtBG1Hm5bK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "lr = 3e-4\n",
        "n_dim = 64\n",
        "in_dim = 28*28*1\n",
        "batch_size = 64\n",
        "num_epochs = 100"
      ],
      "metadata": {
        "id": "zfUsBd0g-n9o"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "disc = Discriminator(in_dim).to(device)\n",
        "gen = Generator(n_dim, in_dim).to(device)\n",
        "fixed_noise ="
      ],
      "metadata": {
        "id": "a14lTZK5_CnF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "h5hrA8l5AMbK"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}