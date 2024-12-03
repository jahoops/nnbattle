import torch


def test_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        device = torch.device("cuda")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print(f"Tensor on GPU: {x}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    test_cuda()
