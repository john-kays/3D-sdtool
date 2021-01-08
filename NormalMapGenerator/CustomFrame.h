#pragma once

#include <QFrame>
#include <QPixmap>

class CustomFrame : public QFrame
{
    Q_OBJECT

public:
    CustomFrame(QObject *parent);
    void SetImage(const QString& imagePath);
    void SetImage(const QImage& image);
    ~CustomFrame();

    inline QString getPath() { return path; }
protected:
    virtual void paintEvent(QPaintEvent* /*e*/);
private:
    QPixmap mPixMap;
    QString path;
};
