import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        ls = [1, 2, 3, 4, 5]
        for i in range(len(ls)):
            ls[i] = ls[i] + 1
        print(ls)


if __name__ == '__main__':
    unittest.main()
