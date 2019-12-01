extern "C" {
#include <cairo/cairo.h>
#include <gtk-3.0/gtk/gtk.h>
}
#include <cinttypes>
#include <iostream>
#include <memory>
#include <fstream>
// cairo for drawing
// gtk for widgets

cairo_surface_t *image = nullptr;

extern uint8_t *GetImage(size_t *height, size_t *width);

static bool on_draw_event(GtkWidget *widget, cairo_t *cr, gpointer user_data) {
  cairo_set_source_surface(cr, image, 10, 10);
  /*
  cairo_set_source_rgb(cr, 10, 12, 12);
  cairo_select_font_face(cr, "Purisa", CAIRO_FONT_SLANT_NORMAL,
                         CAIRO_FONT_WEIGHT_NORMAL);

  cairo_set_font_size(cr, 89.5f);

  cairo_move_to(cr, 132.5f, 132.5f);
  cairo_show_text(cr, "Kaido");
  */

  cairo_paint(cr);

  return false;
}

static bool on_click(GtkWidget *widget, gpointer user_data) {
  std::cout << "Clicked" << std::endl;
  return false;
}

int main(int argc, char *argv[]) {
  size_t height = 0;
  size_t width = 0;
  auto raw_image = GetImage(&height, &width);

  

  std::unique_ptr<uint32_t[]> cairo_image{new uint32_t[height * width]};

  // ideally 32-bit aligned for RGB24
  auto stride =
      cairo_format_stride_for_width(cairo_format_t::CAIRO_FORMAT_RGB24, width);
  for (int i = 0, j = 0; i < height * width * 3; i += 3, j++) {
    cairo_image[j] = ((static_cast<uint32_t>(raw_image[i])) << 16) |
                     (static_cast<uint32_t>(raw_image[i + 1]) << 8) |
                     static_cast<uint32_t>(raw_image[i + 2]);
    
  }

  

  std::cout << std::endl << stride << std::endl;
  image = cairo_image_surface_create_for_data(
      reinterpret_cast<uint8_t *>(cairo_image.get()),
      cairo_format_t::CAIRO_FORMAT_RGB24, width, height, stride);

  gtk_init(&argc, &argv);

  GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  auto button = gtk_button_new_with_label("Click");
  gtk_widget_set_size_request(button, 70, 30);

  GtkWidget *darea = gtk_drawing_area_new();
  auto grid = gtk_grid_new();
  gtk_widget_set_size_request(darea, width, height);

  gtk_grid_attach(GTK_GRID(grid), button, 0, 0, 1, 1);
  gtk_grid_attach(GTK_GRID(grid), darea, 1, 0, 1, 1);

  gtk_container_add(GTK_CONTAINER(window), grid);

  g_signal_connect(G_OBJECT(darea), "draw", G_CALLBACK(on_draw_event), NULL);
  g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(on_click), NULL);

  g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

  gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
  gtk_window_set_default_size(GTK_WINDOW(window), width, height);
  gtk_window_set_title(GTK_WINDOW(window), "Barracuda");
  GError *error = nullptr;
  gtk_window_set_icon_from_file(GTK_WINDOW(window), "icon.png", &error);

  if (error == nullptr) {
    std::cout << "No Errors Occured\n";
  }
  gtk_window_set_resizable(GTK_WINDOW(window), true);

  gtk_widget_show_all(window);

  gtk_main();

  cairo_surface_destroy(image);

  return 0;
}
