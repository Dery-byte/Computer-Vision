# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

### THE PSNR AND THE COMPRESSION RATION ARE SHOWN IN THE CONSOLE.

def PSNR(name):
    import cv2
    import numpy as np
    # Load the original image
    img_original = cv2.imread("image5.jpg")
    # Apply JPEG compression to the image
    jpeg_quality = 50
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, img_jpeg = cv2.imencode('.jpg', img_original, encode_param)
    # Decode the JPEG compressed image
    img_decompressed = cv2.imdecode(img_jpeg, cv2.IMREAD_COLOR)

    # Calculate the mean squared error (MSE) between the original and decompressed images
    mse = np.mean((img_original - img_decompressed) ** 2)
    # Calculate the maximum possible pixel value of the image
    max_pixel_value = 255

    # Calculate the PSNR using the formula: PSNR = 20 * log10(MAX_I) / 10 * log10(MSE)
    psnr = 20 * np.log10(max_pixel_value) / 10 * np.log10(mse)

    # Print the PSNR value
    print(f"PSNR: {psnr} dB")
    img_decompressed_resized = cv2.resize(img_decompressed, (600, 600), 1)
    img_original_resized = cv2.resize(img_original, (600, 600), 1)

    cv2.imshow("compressed image",img_decompressed_resized)
    cv2.imshow("Original Image", img_original_resized)

    #Calculating the compression ratio
    original_image_size= img_original.shape
    compressed_image_size=img_decompressed_resized.shape
    print("The original Size is :",original_image_size);
    print("The compressed image size is:",compressed_image_size);

    CR= (original_image_size[0]* original_image_size[1]* original_image_size[2])/(compressed_image_size[0] * compressed_image_size[1] * compressed_image_size[2]);
    print("The compression Ration is :",CR)

    cv2.imwrite("img_decompressed.jpg", img_decompressed_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    PSNR('PyCharm')
