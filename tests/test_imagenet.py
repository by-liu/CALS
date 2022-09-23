import unittest

from calibrate.data.imagenet import build_train_val_dataset


class TestImagenet(unittest.TestCase):
    def test_imagenet(self):
        data_root = "./data/ImageNet"
        train_dataset, val_dataset = build_train_val_dataset(data_root, use_mysplit=False)

        print(train_dataset)
        for i in range(10):
            img, target = train_dataset[i]
            print(img.shape, target)

        print(val_dataset)
        for i in range(10):
            img, target = val_dataset[i]
            print(img.shape, target)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
