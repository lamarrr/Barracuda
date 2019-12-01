#include <QtGui/QIcon>
#include <QtGui/QImage>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>

char* name = "Kaido";
char* names[] = {name};

int main(int argc, char** argv) {
  int count = 1;
  QApplication app{count, names};
  uint8_t data[] = {0xff, 0x00, 0x00, 0xfa, 0xfa, 0x00, 0xf1, 0xee, 0xef};
  QImage image{"/home/lamar/Desktop/kaido.jpeg"};
  QLabel label;
  QIcon icon{"/home/lamar/Desktop/barracuda/icon.png"};
  app.setWindowIcon(icon);
  label.setPixmap(QPixmap::fromImage(image));
  label.show();
  return app.exec();
}
