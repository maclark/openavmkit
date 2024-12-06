import unittest
from openavmkit.module import hello_world

class TestHelloWorld(unittest.TestCase):
    def test_hello_world(self):
        """
        Test that hello_world() returns the expected string.
        """
        self.assertEqual(hello_world(), "Hello, world!")

if __name__ == "__main__":
    unittest.main()
