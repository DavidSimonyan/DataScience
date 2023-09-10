import torch

def PrintGPUSupport():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))


if __name__ == '__main__':
    PrintGPUSupport()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
