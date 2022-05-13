import torch


def gradient_check(model_param_list, threshold):
    return sum(map(lambda x: torch.norm(x.grad.data), model_param_list)) < threshold


if __name__ == '__main__':
    test = torch.Tensor([[1,2],[3,4],[5,6]])
    print(torch.norm(test))