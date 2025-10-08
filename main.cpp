#include "bitmap.h"
#include <cmath>
#include <iostream>

int main() {

    std::cout << "Trying to detect edges" << std::endl;
    Bitmap image = load_image_grayscale("output_display/image.PNG");
    // Apply Gaussian blur
    Bitmap blurred_image = gaussian_blur(image, 11, 1.0f);
    save_bitmap(blurred_image, "output_display/blurred_image.png");
    

    FloatMap sobel_vertical_kernel = get_sobel_kernel(true);
    FloatMap sobel_vertical_image = apply_kernel(image, sobel_vertical_kernel);

    Bitmap visual = create_bitmap_from_floatmap(sobel_vertical_image);


    save_bitmap(visual, "output_display/sobel_vertical.png");

    FloatMap sobel_horizontal_kernel = get_sobel_kernel(false);
    FloatMap sobel_horizontal_image = apply_kernel(image, sobel_horizontal_kernel);
    Bitmap visual_horizontal = create_bitmap_from_floatmap(sobel_horizontal_image);
    save_bitmap(visual_horizontal, "output_display/sobel_horizontal.png");


    // magnitude = sqrt(Gx^2 + Gy^2)
    FloatMap magnitude(sobel_vertical_image.width, sobel_vertical_image.height);
    for (int y = 0; y < magnitude.height; y++) {
        for (int x = 0; x < magnitude.width; x++) {
            float gx = sobel_horizontal_image.data[y][x];
            float gy = sobel_vertical_image.data[y][x];
            magnitude.data[y][x] = std::sqrt(gx * gx + gy * gy);
        }
    }

    Bitmap magnitude_visual = create_bitmap_from_floatmap(magnitude);
    save_bitmap(magnitude_visual, "output_display/magnitude.png");

    FloatMap direction = FloatMap(sobel_vertical_image.width, sobel_vertical_image.height);
    for (int y = 0; y < direction.height; y++) {
        for (int x = 0; x < direction.width; x++) {
            float gx = sobel_horizontal_image.data[y][x];
            float gy = sobel_vertical_image.data[y][x];
            direction.data[y][x] = std::atan2(gy, gx);
        }
    }
    Bitmap direction_visual = create_bitmap_from_floatmap(direction);
    save_bitmap(direction_visual, "output_display/direction.png");

    std::cout << "Saved everything to output_display/" << std::endl;


    return 0;
}