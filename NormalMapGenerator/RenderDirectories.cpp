#include "RenderDirectories.h"
#include <QDir>

#define FILE_PATH_SEPARATOR_STR "//"

RenderDirectories::RenderDirectories()
{
    m_sBasePath = getenv(("USERPROFILE"));
    m_sBasePath += FILE_PATH_SEPARATOR_STR;
    m_sBasePath += "AppData";
    m_sBasePath += FILE_PATH_SEPARATOR_STR;
    m_sBasePath += "Roaming";
    m_sBasePath += FILE_PATH_SEPARATOR_STR;
    m_sBasePath += "mnm";
    if (!QDir(m_sBasePath.c_str()).exists())
        QDir().mkdir(m_sBasePath.c_str());
    m_sBasePath += FILE_PATH_SEPARATOR_STR;
    m_sBasePath += "RenderingEngine";
 //   m_sBasePath += FILE_PATH_SEPARATOR_STR;
    if (!QDir(m_sBasePath.c_str()).exists())
        QDir().mkdir(m_sBasePath.c_str());

    
}

RenderDirectories::~RenderDirectories()
{

}

void RenderDirectories::GetNormalTexturePath(std::string& sPath)
{
    sPath = m_sBasePath + FILE_PATH_SEPARATOR_STR + "NormalMap.png";
}

void RenderDirectories::GetRoughnessTexturePath(std::string& sPath)
{
    sPath = m_sBasePath + FILE_PATH_SEPARATOR_STR + "Roughness.png";
}