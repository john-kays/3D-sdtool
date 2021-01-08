#include "MainWindow.h"
#include <QFileDialog>
#include "normalmapgenerator.h"
#include "RenderDirectories.h"

static QString GetFileNameFromDlg(QWidget* parent)
{
    QString filter = " Files (*.jpeg, *.jpg) ;;";
    return QFileDialog::getOpenFileName(parent, "Select a file :", QDir::homePath(), filter);
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    showMaximized();
    InitMenu();
}

void MainWindow::InitMenu()
{
    QMenu* fileMenu = new QMenu(this->menuBar());
    fileMenu->setTitle("File");
    QAction* pAction = new QAction("Open", fileMenu);
    pAction->setShortcut(QKeySequence("Ctrl + O"));
    fileMenu->addAction(pAction);
    QObject::connect(pAction, SIGNAL(triggered()),
        this, SLOT(OnLoadImage()));

    QAction* pSaveNormalAction = new QAction("Save Normal Map", fileMenu);
    fileMenu->addAction(pSaveNormalAction);
    QObject::connect(pSaveNormalAction, SIGNAL(triggered()),
        this, SLOT(OnSaveNormalImage()));


    this->menuBar()->addMenu(fileMenu);
}

void MainWindow::OnLoadImage()
{
    QString selectedFile = GetFileNameFromDlg(this);
    if (selectedFile.length() == 0)
        return;

    ui.baseFrame->SetImage(selectedFile);

    RenderDirectories renderDir;
    NormalmapGenerator normalmapGenerator;
    QImage normalmap = normalmapGenerator.calculateNormalmap(QImage(selectedFile), 2.0, true);
    std::string path;
    renderDir.GetNormalTexturePath(path);
    normalmap.save(path.c_str());
    ui.normalFrame->SetImage(path.c_str());
}

void MainWindow::OnSaveNormalImage()
{
    QString fileName = QFileDialog::getSaveFileName(this,
        tr("Save"), "",
        tr("(*.jpeg);;All Files (*)"));
    if (fileName.length() == 0)
        return;
    QImage normalImage(ui.normalFrame->getPath());
    normalImage.save(fileName);
}
