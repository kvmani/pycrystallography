#setup.py
from cx_Freeze import setup, Executable
import os.path
import glob
PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')


dataDirs=[r'..\data\structureData',
         r'..\data\SADCalib',
         r'..\data\defaults',]

additinalFiles =[]
for dataDir in dataDirs:
    files = glob.glob(os.path.join(dataDir,"*.*"))
    #files2 = glob.glob(dataDir+r"/"+"*.json")
    
    if files is not None :
        if len(files)>0:
            for file in files:
                additinalFiles.append(file)
print(additinalFiles)

options = {
    'build_exe': {
        'packages': ["os","sys","ctypes","win32con"],
        'include_files':[
            os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tk86t.dll'),
            os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tcl86t.dll'),
         ].extend(additinalFiles),
        'include_msvcr': True,
    },
}

setup(
    name = "SAEDSolver",
    version = "1.0.0",
    options = options,
    executables = [Executable("examplePlotGUI.py",base="Win32GUI")]
    )