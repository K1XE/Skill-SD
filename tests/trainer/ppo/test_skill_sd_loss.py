import pytest
import torch

from verl.trainer.ppo.skill_sd_loss import compute_sdl_loss


@pytest.mark.parametrize("mode", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"])
def test_matching_teacher_has_zero_loss_and_gradient(mode):
    student = torch.tensor([[-0.2, -1.0], [-0.8, -0.3]], requires_grad=True)
    teacher = student.detach().clone().requires_grad_()
    old = torch.full_like(student, -0.5, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    loss = compute_sdl_loss(student, teacher, old, mask, mode)
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(student.grad, torch.zeros_like(student))
    assert teacher.grad is None
    assert old.grad is None


def test_distillation_moves_student_toward_teacher_and_ignores_padding():
    student = torch.tensor([[-0.2, -1.0, -0.1]], requires_grad=True)
    teacher = torch.tensor([[-0.8, -0.3, -4.0]], requires_grad=True)
    old = torch.full_like(student, -0.5, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    loss = compute_sdl_loss(student, teacher, old, mask)
    loss.backward()

    assert loss.item() > 0.0
    assert student.grad[0, 0] > 0.0
    assert student.grad[0, 1] < 0.0
    assert student.grad[0, 2] == 0.0
    assert teacher.grad is None
    assert old.grad is None
    unpadded = compute_sdl_loss(student[:, :2], teacher[:, :2], old[:, :2], mask[:, :2])
    torch.testing.assert_close(loss, unpadded)
