import os
import ctypes

dll_path = r"C:\Users\ivaku\Documents\Script_2.0\J_0_test\mconf\mconf.src\bin\mconf_matlab64.dll"

if os.path.exists(dll_path):
    print("DLL найден!")
    try:
        ctypes.WinDLL(dll_path)
        print("DLL загружен успешно!")
    except Exception as e:
        print(f"Ошибка загрузки DLL: {e}")
else:
    print("DLL НЕ найден!")
