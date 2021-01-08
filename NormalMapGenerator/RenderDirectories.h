#pragma once
#include <string>
class RenderDirectories
{
public:
    RenderDirectories();
    ~RenderDirectories();

    void GetNormalTexturePath(std::string& sPath);
    void GetRoughnessTexturePath(std::string& sPath);
private:
    std::string m_sBasePath;
};

