import unittest
import torch
from SignalTorch import (
	MoveLastToFirst,
	MoveFirstToLast,
	roll, shft,
	roll_mat,
	shft_mat,
	conv_t,
	win_t
)

class TestSignalProcessing(unittest.TestCase):
    def setUp(self):
        self.tensor = torch.arange(1, 10).float()
        self.filter = torch.tensor([0.1, 0.2, 0.3])

    def test_MoveLastToFirst(self):
        result = MoveLastToFirst(self.tensor)
        expected = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.])
        self.assertTrue(torch.allclose(result, expected))

    def test_MoveFirstToLast(self):
        result = MoveFirstToLast(self.tensor)
        expected = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.])
        self.assertTrue(torch.allclose(result, expected))

    def test_roll(self):
        result = roll(self.tensor, 2)
        expected = torch.tensor([8., 9., 1., 2., 3., 4., 5., 6., 7.])
        self.assertTrue(torch.allclose(result, expected))

    def test_shft(self):
        result = shft(self.tensor, 2)
        expected = torch.tensor([0., 0., 1., 2., 3., 4., 5., 6., 7.])
        self.assertTrue(torch.allclose(result, expected))

    def test_roll_mat(self):
        result = roll_mat(self.tensor, 3)
        expected = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                 [9., 1., 2., 3., 4., 5., 6., 7., 8.],
                                 [8., 9., 1., 2., 3., 4., 5., 6., 7.]])
        self.assertTrue(torch.allclose(result, expected))

    def test_shft_mat(self):
        result = shft_mat(self.tensor, 3)
        expected = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                 [0., 0., 0., 0., 0., 0., 0., 1., 2.],
                                 [0., 0., 0., 0., 0., 0., 1., 2., 3.]])
        self.assertTrue(torch.allclose(result, expected))

    def test_conv_t(self):
        result = conv_t(self.tensor, self.filter)
        expected = torch.tensor([2.6, 3.3, 4. , 4.7, 5.4, 6.1, 6.8, 7.5, 8.2])
        self.assertTrue(torch.allclose(result, expected))

    def test_win_t(self):
        result = win_t(self.tensor)
        expected = torch.tensor([0.0, 0.14644661, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertTrue(torch.allclose(result, expected))

if __name__ == '__main__':
    unittest.main()
