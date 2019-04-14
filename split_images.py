import glob
import image_slicer
from PIL import Image

dataset_path = 'dataset/600_chloride/*.tif'
save_path_cropped_images = 'dataset/512X512_CROPPED_600_CHLORIDE/'
images = glob.glob(dataset_path)

for index,image in enumerate(images):
    img = Image.open(image)
    area = (0, 0, 512, 512)
    cropped_img = img.crop(area).save(save_path_cropped_images+str(index)+'.tif',format=None)
    
cropped_images_path = 'dataset/512X512_CROPPED_600_CHLORIDE/*.tif'
cropped_images = glob.glob(cropped_images_path)

tiles_save_path = 'dataset/32x32_600_CHLORIDE/'

tiles_save_path_test = 'dataset/32x32_600_CHLORIDE_TEST/'


# for index,image in enumerate(cropped_images):
#     tiles = image_slicer.slice(image, 16, save=False)
#     image_slicer.save_tiles(tiles, directory=tiles_save_path,\
#                             prefix=str(index), format='png')

for index,image in enumerate(cropped_images):
    tiles = image_slicer.slice(image, 16, save=False)
    image_slicer.save_tiles(tiles, directory=tiles_save_path_test,\
                            prefix=str(index), format='png')

