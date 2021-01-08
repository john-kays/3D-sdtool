#include "CustomFrame.h"
#include <QPaintEvent>
#include <QPainter>

CustomFrame::CustomFrame(QObject *parent)

{
}

CustomFrame::~CustomFrame()
{
}

void CustomFrame::SetImage(const QString& imagePath)
{
    path = imagePath;
    mPixMap.load(imagePath);
}

void CustomFrame::SetImage(const QImage& image)
{
    mPixMap.fromImage(image);
}

void CustomFrame::paintEvent(QPaintEvent* e)
{
//     QPainter p(this);
//     p.drawPixmap(0, 0, width(), height(), mPixMap).scaled;
    QPainter painter(this);
    painter.drawPixmap(0, 0, mPixMap.scaled(size()));
    QWidget::paintEvent(e);
}