/**
 * @file better.cc
 * @author Basit Ayantunde (rlamarrr@gmail.com)
 * @brief
 * @version 0.1.0
 * @date 2019-11-30
 *
 * @copyright Copyright (c) 2019
 *
 */

#include <gdkmm.h>
#include <gtkmm.h>

class BarArea : public Gtk::DrawingArea {
 public:
  bool on_draw(const Cairo::RefPtr<Cairo::Context>& cr) override {
    // if (!m_image) {
    //  return false;
    // }

    // Cairo::RefPtr<Gdk::PixBuf> pixbuf;
    // Gdk::Cairo::set_source_pixbuf(cr, m_image, 0, 0);
    cr->rectangle(0, 0, 100, 100);
    cr->fill();
    

    // cr->paint();
    return true;
  }

  ~BarArea() override {}
  // create from resource
  // Glib::RefPtr<Gdk::Pixbuf> m_image;
};

int main(int argc, char* argv[]) {
  auto app = Gtk::Application::create("Barracuda");

  Gtk::Window window;
  window.set_default_size(800, 800);

  BarArea drawing_area;
  window.add(drawing_area);
  drawing_area.show();

  return app->run(window);
}
