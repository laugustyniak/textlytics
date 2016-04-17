# import unittest
import unittest2 as unittest


class MyTestCase(unittest.TestCase):
	def test_something(self):
		self.assertEqual(True, True)


if __name__ == '__main__':
	unittest.main()
