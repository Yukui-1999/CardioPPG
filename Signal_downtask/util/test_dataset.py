import torch
import unittest
from util.dataset import ECGDataset  # 请根据实际情况修改这个导入语句

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.data_path = "/home/dingzhengyao/Work/ECG_CMR/mae/mae-ba56dd91a7b8db544c1cb0df3a00c5c8a90fbb65/ecg_data_array_val.pt"  # 请替换为实际的数据文件路径
        # self.labels_path = "/path/to/your/labels.pt"  # 请替换为实际的标签文件路径
        self.dataset = ECGDataset(data_path=self.data_path)

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.dataset.data))

    def test_getitem(self):
        data,lable,lable_mask = self.dataset[0]
        print(data.shape)
        self.assertTrue(torch.equal(data, self.dataset.data[0]))
        # self.assertTrue(torch.equal(label, self.dataset.labels[0]))

if __name__ == "__main__":
    unittest.main()