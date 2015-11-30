# coding: utf-8

from JCMpython import *


# Globals
tsep = '\n'+80*'='+'\n'
psep = 60*'-'
uol = 1e-9 #m
a = 1000. #nm


class TestResult(object):
    
    before = tsep+'Test: {0}'+tsep
    after = tsep+'Finished'+tsep
    
    def __init__(self, testName = 'test', description = ''):
        self.testName = testName
        self.description = description
        self.notes = ''
        self.results = []
        self.start()
    
    def start(self):
        if self.description:
            print self.before.format(self.testName+'\n\n'+self.description)
        else:
            print self.before.format(self.testName)
    
    def add(self, text, dump = True):
        self.notes += text
        if dump: self.dump()
    
    def dump(self):
        print self.notes
        self.notes = ''
    
    def result(self, res, finish = True):
        self.add('Success? ... ' + str(res))
        self.results.append(res)
        if finish: self.finish()
        
    def conclusion(self, finish = True):
        self.add('Passed all tests? ... ' + str(all(self.results)))
        if finish: self.finish()
    
    def section(self, text):
        print psep
        print text
        print psep
    
    def finish(self):
        print self.after


def unitTest():
    
    #-
    t1 = TestResult('Initializing blochVectors')
    Gamma = blochVector( 0., 
                         0., 
                         0., 
                         'Gamma',
                         isGreek = True )
    
    M     = blochVector( 0., 
                         2.*np.pi / a / uol / np.sqrt(3.), 
                         0., 
                         'M' )
    
    K     = blochVector( 2.*np.pi / a / uol / 3.,
                         2.*np.pi / a / uol / np.sqrt(3.),
                         0.,
                         'K' )
    res = isinstance(Gamma, blochVector) and \
          isinstance(M, blochVector) and \
          isinstance(K, blochVector)
    t1.result(res)
    
    
    #-
    t2 = TestResult('Creating a BrillouinPath')
    path = [M, K, Gamma, M]
    brillouinPath = BrillouinPath( path )
    res = isinstance(brillouinPath, BrillouinPath)
    t2.result(res)
    
    
    #-
    t3 = TestResult('Initializing a Bandstructure and status info output')
    # clean
    rmtree('bsUnitTest')
    t3.add('Saving:')
    bs = Bandstructure(storageFolder = 'bsUnitTest', dimensionality = 3, 
                       nBands=8, brillouinPath=brillouinPath, nKvals=200, 
                       polarizations = ['TE','TM'])
    t3.add('\nLoading:')
    del bs
    bs = Bandstructure(storageFolder = 'bsUnitTest')
    t3.add('\nStatus ouput:')
    bs.statusInfo()
    t3.finish()
    
    
    #-
    t4 = TestResult('Adding results to a Bandstructure-instance')
    kIdx = 15
    bandIdx = 2
    
    # Test cases
    singleSolve = getSingleKdFrame(kIdx, band=bandIdx)
    test_array = np.arange(len(bandColumns))
    test_list = test_array.tolist()
    singleSolve.iloc[0] = test_array
    test_dframe = singleSolve
    test_dict1 = singleSolve.to_dict()
    test_dict2 = {tdk[1]:test_dict1[tdk][kIdx] for tdk in test_dict1.keys()}
    tuples = [(dKey, test_dict2[dKey]) for dKey in test_dict2.keys()]
    
    print test_dict1
    print test_dict2
    
    # Validation
    def check1(d):
        d.loc[15,bname(bandIdx)] = np.nan
        return np.all(d.loc[15,bname(bandIdx)].isnull())
    def check2(d):
        return np.all( test_array == d.loc[kIdx, bname(bandIdx)].values )
    
    t4.section('Valid tests')
    
    t4.add('Array')
    c1 = check1(bs.data)
    bs.addResults(array=test_array, k=kIdx, band=bandIdx)
    c2 = check2(bs.data)
    t4.result(c1 and c2, False)
    
    t4.add('List')
    c1 = check1(bs.data)
    bs.addResults(array=test_list, k=kIdx, band=bandIdx)
    c2 = check2(bs.data)
    t4.result(c1 and c2, False)
    
    t4.add('DFrame')
    c1 = check1(bs.data)
    bs.addResults(test_dframe)
    c2 = check2(bs.data)
    t4.result(c1 and c2, False)
    
    t4.add('Dict1')
    c1 = check1(bs.data)
    bs.addResults(rDict=test_dict1)
    c2 = check2(bs.data)
    t4.result(c1 and c2, False)
    
    t4.add('Dict2')
    c1 = check1(bs.data)
    bs.addResults(rDict=test_dict2, k=kIdx, band=bandIdx)
    c2 = check2(bs.data)
    t4.result(c1 and c2, False)
    
    t4.add('Tuples')
    c1 = check1(bs.data)
    for i,tt in enumerate(tuples):
        bs.addResults(singleValueTuple=tt, k=kIdx, band=bandIdx,
                      save=i+1==len(tuples))
    c2 = check2(bs.data)
    t4.result(c1 and c2, False)
    t4.conclusion(False)
    
    t4.section('Failing tests')
    t4.add('\ntest_dict2 without band and k')
    c1 = check1(bs.data)
    bs.addResults(rDict=test_dict2)
    c2 = check2(bs.data)
    t4.result(c1 and c2, False)
    
    t4.add('\ntest_array of wrong size')
    c1 = check1(bs.data)
    bs.addResults(array=test_array[:-1], k=kIdx, band=bandIdx)
    c2 = check2(bs.data)
    t4.result(c1 and c2, False)
    
    t4.add('\ntuples without band and k')
    c1 = check1(bs.data)
    for tt in tuples:
        bs.addResults(singleValueTuple=tt)
    c2 = check2(bs.data)
    t4.result(c1 and c2, False)
    t4.finish()
    
    
if __name__ == '__main__':
    unitTest()