from self_driving_car_modules.line_extraction import * 

if __name__=='__main__':
    #test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
    # show_images(test_images)
    # show_images(list(map(select_rgb_white_yellow, test_images)))
    # show_images(list(map(convert_hsv, test_images)))
    #show_images(list(map(convert_hls, test_images)))

    test_images = [plt.imread(path) for path in glob.glob('../udacity_datasets/*.jpg')]
    show_images(test_images)
    rgb_images = (list(map(select_rgb_white_yellow, test_images)))
    # show_images(rgb_images)
    hsv_images = list(map(convert_hsv, rgb_images))
    
    hls_images =list(map(convert_hls, rgb_images))
    # show_images(hls_images)
    white_yellow_images = list(map(select_white_yellow, test_images))
    # show_images(white_yellow_images)
    gray_images = list(map(convert_gray_scale, white_yellow_images))
    # show_images(gray_images)
    blurred_images = list(map(lambda image: apply_smoothing(image), gray_images))
    # show_images(blurred_images)
    
    edge_images = list(map(lambda image: detect_edges(image), blurred_images))

    # show_images(edge_images)
    roi_images = list(map(select_region, edge_images))
    # show_images(roi_images)
    list_of_lines = list(map(hough_lines, roi_images))
    print(list_of_lines)
    line_images = []
    for image, lines in zip(test_images, list_of_lines):
        line_images.append(draw_lines(image, lines))
    
    show_images(line_images)
    lane_images = []
    for image, lines in zip(test_images, list_of_lines):
        lane_images.append(draw_lane_lines(image, lane_lines(image, lines)))

    
    show_images(lane_images)