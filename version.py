import torch
if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())