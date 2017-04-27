import unittest

class DBTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testConnection(self):
        raise NotImplementedError

    def testInsert(self):
        raise NotImplementedError

    def testSelectNoArgument(self):
        raise NotImplementedError

    def testSelectOneArgument(self):
        raise NotImplementedError

    def testSelectMultiArgument(self):
        raise NotImplementedError





if __name__ == '__main__':
    unittest.main()
