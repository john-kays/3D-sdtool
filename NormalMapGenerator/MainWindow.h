#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_NormalMapGenerator.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = Q_NULLPTR);


protected slots:

    void OnLoadImage();
    void OnSaveNormalImage();
private:

    void InitMenu();
private:
    Ui::NormalMapGeneratorClass ui;
};
