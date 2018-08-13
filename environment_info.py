"""Outputs info on the environment in which pypmj runs and other specific
data useful for bug filing to an output text file.

Authors : Carlo Barth

"""

import importlib
import os
import platform
from shutil import rmtree
import sys

# For final clean-up, remember if the logs folder exists in this working
# directory. If so, also remember its content.
LOG_FOLDER_EXISTS = os.path.isdir('logs')
if LOG_FOLDER_EXISTS:
    LOG_FOLDER_CONTENTS = os.listdir('logs')

FMT = '{0:20s}:  {1}\n'
def fmt(key_, val_):
    return FMT.format(key_, val_)

SEP = '\n'+80*'='+'\n'

def main():
    fout = open('pypmj_env_info.log', 'w')
    
    fout.write(SEP)
    fout.write(' Platform and version data')
    fout.write(SEP+'\n')
    
    fout.write(fmt('platform', platform.platform()))
    fout.write(fmt('python version', sys.version))
    
    # Check dependency imports and versions
    dependencies = ('numpy', 'pandas', 'scipy', 'tables')
    dep_versions = {}
    missing_dependencies = []
    for d in dependencies:
        try:
            module = importlib.import_module(d)
            dep_versions[d] = module.__version__
        except ImportError as e:
            missing_dependencies.append((d, e))
    for d in dep_versions:
        fout.write(fmt(d+' version', dep_versions[d]))
    if missing_dependencies:
        for md in missing_dependencies:
            d, e = md
            fout.write(fmt(d+'->ImportError', e))
        fout.write('\nLeaving.')
        fout.close()
        return
    
    try:
        import pypmj as jpy
    except Exception as e:
        fout.write(fmt('pypmj->ImportError', e))
        fout.write('\nLeaving.')
        fout.close()
        return
    
    try:
        with open(jpy._config.config_file, 'r') as f:
            conf_content = f.read()
        fout.write(fmt('pypmj version', jpy.__version__))
        fout.write(fmt('JCMsuite version', jpy.__jcm_version__))
        fout.write(SEP)
        fout.write(' The config file')
        fout.write(SEP+'\n')
        fout.write(conf_content)
        fout.write(SEP)
        fout.write(' jcm_license_info')
        fout.write(SEP)
        fout.write('\n{}\n'.format(jpy.jcm_license_info(False,True)))
        fout.write(SEP)
        fout.close()
    except:
        fout.close()
    
    # Clean up
    if LOG_FOLDER_EXISTS:
        if not os.path.isdir('logs'):
            return
        contents = os.listdir('logs')
        for c in contents:
            cpath = os.path.join('logs', c)
            if not c in LOG_FOLDER_CONTENTS:
                if os.path.isdir(cpath):
                    rmtree(os.path.join('logs', c))
                else:
                    os.remove(cpath)
    else:
        if os.path.isdir('logs'):
            rmtree('logs')



if __name__ == '__main__':
    main()
