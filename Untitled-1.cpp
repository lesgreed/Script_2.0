#include <windows.h>
#include <iostream>
#include <fstream>

BOOL APIENTRY DllMain(HMODULE hModule,
                      DWORD  ul_reason_for_call,
                      LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

// Function to create and save a DLL file
void createDllFile(const std::string& filename) {
    std::ofstream dllFile(filename, std::ios::binary);
    if (dllFile.is_open()) {
        // Writing minimal DLL structure (empty, just as a placeholder)
        char emptyData[512] = {0};
        dllFile.write(emptyData, sizeof(emptyData));
        dllFile.close();
        std::cout << "DLL file created: " << filename << std::endl;
    } else {
        std::cerr << "Failed to create DLL file." << std::endl;
    }
}

int main() {
    createDllFile("EmptyX64Dll.dll");
    return 0;
}