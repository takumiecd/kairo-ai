import unittest

from train.train import select_device
from train.train import split_dataset


class TrainEntrypointTest(unittest.TestCase):
    def test_selects_cpu_device(self):
        self.assertEqual(str(select_device("cpu")), "cpu")

    def test_split_dataset_creates_validation_subset(self):
        dataset = list(range(10))

        train_dataset, valid_dataset = split_dataset(dataset, validation_ratio=0.2, seed=0)

        self.assertEqual(len(train_dataset), 8)
        self.assertEqual(len(valid_dataset), 2)

    def test_split_dataset_rejects_invalid_ratio(self):
        with self.assertRaises(ValueError):
            split_dataset([1, 2, 3], validation_ratio=1.0, seed=0)


if __name__ == "__main__":
    unittest.main()
