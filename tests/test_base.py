"""Basic unit tests for jcmpython.

Authors : Carlo Barth

"""

import jcmpython as jpy
import logging
import unittest
logger = logging.getLogger(__name__)


class Test_JCMbasics(unittest.TestCase):
    
    def test_print_info(self):
        jpy.jcm.info()



if __name__ == '__main__':
    logger.info('This is test_base.py')
    
    suites = [
        unittest.TestLoader().loadTestsFromTestCase(Test_JCMbasics)
    ]
    
    for suite in suites:
        unittest.TextTestRunner(verbosity=2).run(suite)