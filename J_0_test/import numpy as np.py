import ctypes
import os


dll_path = "empty.dll"  

if os.path.exists(dll_path):
    print(f"File {dll_path} was found!")
    try:
        full_path = os.path.join(os.getcwd(),dll_path)
        #$full_path = ":\\share.ipp-hgw.mpg.de\documents\ivaku\Documents\GitHub\Script_2\empty.dll"
        full_path = r"C:\Users\ivaku\Documents\GitHub\Script_2\empty.dll"
        print(full_path)
        mconf = ctypes.CDLL(full_path)
        print("DLL here!")
    except Exception as e:
        print(f"Error with loading DLL: {e}")
else:
    print(f"File {dll_path} wasn't found!")
